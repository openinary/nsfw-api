// Build-time step for the nudity model: converts nsfwjs's MobileNetV2Mid (a
// TensorFlow.js graph model) into an ONNX file, so the server runs both models
// on ONNX Runtime and TensorFlow.js never ships in the image.
//
// Mid rather than nsfwjs's default MobileNetV2: the small one sees skin in any
// blurry peach/orange picture (a flower background with a text bubble scored
// 81% Porn, 0.1% with Mid).
//
// The weights keep nsfwjs's uint8 quantisation (4.4 MB instead of 17 MB);
// the graph dequantises them in float64 exactly like TensorFlow.js does
// (q * scale + min, rounded once to float32), and ONNX Runtime folds that into
// plain constants when the model loads.
// Output: models/nsfw/mobilenet_v2_mid.onnx
import { mkdir, writeFile } from "node:fs/promises";
import { MobileNetV2MidModel } from "nsfwjs/models/mobilenet_v2_mid";

const OUT = "models/nsfw";
const SIZE = 224;

// ---------------------------------------------------------------------------
// Minimal protobuf writer, just enough for onnx.proto.
const varint = (n: number | bigint) => {
  const out = [];
  let v = BigInt.asUintN(64, BigInt(n)); // int64: negatives as two's complement
  do {
    let byte = Number(v & 0x7fn);
    v >>= 7n;
    if (v) byte |= 0x80;
    out.push(byte);
  } while (v);
  return Buffer.from(out);
};
const key = (field: number, wire: number) => varint((field << 3) | wire);
const int = (field: number, n: number | bigint) => Buffer.concat([key(field, 0), varint(n)]);
const bytes = (field: number, buf: Buffer) => Buffer.concat([key(field, 2), varint(buf.length), buf]);
const str = (field: number, s: string) => bytes(field, Buffer.from(s, "utf8"));
const msg = (field: number, parts: Buffer[]) => bytes(field, Buffer.concat(parts));
const float32 = (field: number, f: number) => {
  const b = Buffer.alloc(4);
  b.writeFloatLE(f);
  return Buffer.concat([key(field, 5), b]);
};

const DT = { FLOAT: 1, UINT8: 2, INT64: 7, DOUBLE: 11 };
const ATTR = { FLOAT: 1, INT: 2, STRING: 3, INTS: 7 };

const tensor = (name: string, dataType: number, dims: number[], raw: Buffer) =>
  Buffer.concat([...dims.map((d) => int(1, d)), int(2, dataType), str(8, name), bytes(9, raw)]);

type AttrValue = number | number[] | string;

function attribute(name: string, value: AttrValue) {
  if (typeof value === "string") return Buffer.concat([str(1, name), str(4, value), int(20, ATTR.STRING)]);
  if (Array.isArray(value)) return Buffer.concat([str(1, name), ...value.map((v) => int(8, v)), int(20, ATTR.INTS)]);
  if (Number.isInteger(value)) return Buffer.concat([str(1, name), int(3, value), int(20, ATTR.INT)]);
  return Buffer.concat([str(1, name), float32(2, value), int(20, ATTR.FLOAT)]);
}

const valueInfo = (name: string, dims: number[]) =>
  Buffer.concat([
    str(1, name),
    msg(2, [msg(1, [int(1, DT.FLOAT), msg(2, dims.map((d) => msg(1, [int(1, d)])))])]),
  ]);

// ---------------------------------------------------------------------------
// Graph builder
const nodes: Buffer[] = [];
const initializers: Buffer[] = [];
let counter = 0;
const fresh = (hint: string) => `${hint}_${counter++}`;

function node(op: string, inputs: string[], attrs: Record<string, AttrValue> = {}, name = fresh(op)) {
  const output = name;
  nodes.push(
    Buffer.concat([
      ...inputs.map((i) => str(1, i)),
      str(2, output),
      str(3, name),
      str(4, op),
      ...Object.entries(attrs).map(([k, v]) => msg(5, [attribute(k, v)])),
    ]),
  );
  return output;
}

function constant(name: string, dataType: number, dims: number[], raw: Buffer) {
  initializers.push(tensor(name, dataType, dims, raw));
  return name;
}
const f64 = (name: string, v: number) => {
  const b = Buffer.alloc(8);
  b.writeDoubleLE(v);
  return constant(name, DT.DOUBLE, [], b);
};
const f32 = (name: string, v: number) => {
  const b = Buffer.alloc(4);
  b.writeFloatLE(v);
  return constant(name, DT.FLOAT, [], b);
};

// ---------------------------------------------------------------------------
// Weights: TensorFlow.js manifest + base64 shard bundled in the nsfwjs package.
const modelJson = await MobileNetV2MidModel.modelJson();
const { modelTopology, weightsManifest } = modelJson.default ?? modelJson;
const shards = await Promise.all(MobileNetV2MidModel.weightBundles.map((load) => load()));
const blob = Buffer.concat(shards.map((s) => Buffer.from(s.default ?? s, "base64")));

interface Weight {
  shape: number[];
  q?: { dtype: string; scale?: number; min?: number };
  data: Buffer;
}
const weights = new Map<string, Weight>();
let offset = 0;
for (const w of weightsManifest.flatMap((m) => m.weights)) {
  const count = w.shape.reduce((a, b) => a * b, 1);
  const q = w.quantization;
  if (q && q.dtype !== "uint8") throw new Error(`Unsupported quantization ${q.dtype} for ${w.name}`);
  if (!q && w.dtype !== "float32") throw new Error(`Unsupported dtype ${w.dtype} for ${w.name}`);
  const size = q ? count : count * 4;
  weights.set(w.name, { shape: w.shape, q, data: blob.subarray(offset, offset + size) });
  offset += size;
}
if (offset !== blob.length) throw new Error(`Weight manifest covers ${offset} bytes, shard has ${blob.length}`);

// Reorders a row-major tensor (one element = `width` bytes): out[perm-applied index] = in[index].
function transpose(data: Buffer, shape: number[], perm: number[], width: number) {
  const outShape = perm.map((p) => shape[p]);
  const inStrides = shape.map((_, i) => shape.slice(i + 1).reduce((a, b) => a * b, 1));
  const out = Buffer.alloc(data.length);
  const count = data.length / width;
  const idx = new Array(outShape.length).fill(0);
  for (let o = 0; o < count; o++) {
    let src = 0;
    for (let d = 0; d < outShape.length; d++) src += idx[d] * inStrides[perm[d]];
    data.copy(out, o * width, src * width, (src + 1) * width);
    for (let d = outShape.length - 1; d >= 0 && ++idx[d] === outShape[d]; d--) idx[d] = 0;
  }
  return { data: out, shape: outShape };
}

// Emits a weight, reshaped for ONNX: quantised ones become uint8 + Cast/Mul/Add/Cast.
function weight(name: string, { perm, reshape }: { perm?: number[]; reshape?: number[] } = {}) {
  const w = weights.get(name);
  if (!w) throw new Error(`Missing weight ${name}`);
  weights.delete(name);
  let { data, shape } = w;
  if (perm) ({ data, shape } = transpose(data, shape, perm, w.q ? 1 : 4));
  if (reshape) shape = reshape;
  const id = name.replace(/[^\w]/g, "_");
  if (!w.q) return constant(id, DT.FLOAT, shape, data);
  const q = constant(`${id}_q`, DT.UINT8, shape, data);
  const scaled = node("Mul", [node("Cast", [q], { to: DT.DOUBLE }), f64(`${id}_scale`, w.q.scale!)]);
  return node("Cast", [node("Add", [scaled, f64(`${id}_min`, w.q.min!)])], { to: DT.FLOAT }, id);
}

// ---------------------------------------------------------------------------
// TensorFlow ops -> ONNX (NCHW inside the graph). The converter only knows the
// dozen ops this model uses, and throws on anything else.
interface TfNode {
  name: string;
  op: string;
  input?: string[];
  attr?: Record<string, { s?: string; list?: { i?: string[]; s?: string[] } }>;
}
const b64 = (s?: string) => Buffer.from(s ?? "", "base64").toString();
const ints = (n: TfNode, name: string) => (n.attr?.[name]?.list?.i ?? []).map(Number);
const fused = (n: TfNode) => (n.attr?.fused_ops?.list?.s ?? []).map(b64);

const tensors = new Map<string, string>(); // TF node name -> ONNX tensor name
// An input is either another node's output or a Const, whose weight is emitted here.
const operand = (name: string) => tensors.get(name) ?? weight(name);
const zero = f32("zero", 0);
const six = f32("six", 6);
let output = "";

for (const n of (modelTopology as { node: TfNode[] }).node) {
  const ins = n.input ?? [];
  let y: string;
  switch (n.op) {
    case "Const":
      continue; // emitted by the node that consumes it
    case "Placeholder":
      y = node("Transpose", ["input"], { perm: [0, 3, 1, 2] }, "input_nchw");
      break;
    case "Mul":
    case "Sub":
    case "AddV2":
      y = node(n.op === "AddV2" ? "Add" : n.op, ins.map(operand));
      break;
    case "_FusedConv2D":
    case "FusedDepthwiseConv2dNative": {
      const depthwise = n.op === "FusedDepthwiseConv2dNative";
      const ops = fused(n);
      if (
        b64(n.attr?.data_format?.s) !== "NHWC" ||
        b64(n.attr?.padding?.s) !== "SAME" ||
        ints(n, "dilations").some((d) => d !== 1) ||
        ops[0] !== "BiasAdd" ||
        ops.slice(1).some((o) => o !== "Relu6") ||
        ops.length > 2
      ) {
        throw new Error(`Unsupported convolution ${n.name}`);
      }
      const [kh, kw, cin, cout] = weights.get(ins[1])!.shape;
      // TF kernel [kh, kw, in, out|multiplier] -> ONNX [out, in/group, kh, kw]
      const w = weight(ins[1], {
        perm: depthwise ? [2, 3, 0, 1] : [3, 2, 0, 1],
        reshape: depthwise ? [cin * cout, 1, kh, kw] : undefined,
      });
      y = node("Conv", [operand(ins[0]), w, weight(ins[2])], {
        kernel_shape: [kh, kw],
        strides: ints(n, "strides").slice(1, 3),
        auto_pad: "SAME_UPPER", // TensorFlow's "SAME": the odd pixel goes at the bottom/right
        group: depthwise ? cin : 1,
      });
      if (ops.includes("Relu6")) y = node("Clip", [y, zero, six]);
      break;
    }
    case "AvgPool":
      if (b64(n.attr?.padding?.s) !== "VALID") throw new Error(`Unsupported pooling padding in ${n.name}`);
      y = node("AveragePool", [operand(ins[0])], {
        kernel_shape: ints(n, "ksize").slice(1, 3),
        strides: ints(n, "strides").slice(1, 3),
      });
      break;
    case "Squeeze":
      // Drops the 1x1 spatial dims: the same as flattening NCHW.
      if (ints(n, "squeeze_dims").join() !== "1,2") throw new Error(`Unsupported Squeeze ${n.name}`);
      y = node("Flatten", [operand(ins[0])], { axis: 1 });
      break;
    case "_FusedMatMul":
      if (fused(n).join() !== "BiasAdd") throw new Error(`Unsupported MatMul ${n.name}`);
      y = node("Gemm", ins.map(operand));
      break;
    case "Softmax":
      y = node("Softmax", [operand(ins[0])], { axis: -1 });
      break;
    case "Identity":
      y = output = node("Identity", [operand(ins[0])], {}, "probabilities");
      break;
    default:
      throw new Error(`Unsupported op ${n.op} (${n.name})`);
  }
  tensors.set(n.name, y);
}
if (weights.size) throw new Error(`Unused weights: ${[...weights.keys()].join(", ")}`);

const graph = Buffer.concat([
  ...nodes.map((n) => msg(1, [n])),
  str(2, "nsfwjs_mobilenet_v2_mid"),
  ...initializers.map((t) => msg(5, [t])),
  msg(11, [valueInfo("input", [1, SIZE, SIZE, 3])]),
  msg(12, [valueInfo(output, [1, 5])]),
]);
const model = Buffer.concat([
  int(1, 8), // IR version
  str(2, "openinary/nsfw-api scripts/prepare-nsfw.ts"),
  str(6, "nsfwjs 4.4.0 MobileNetV2Mid (MIT, github.com/infinitered/nsfwjs), converted to ONNX"),
  msg(7, [graph]),
  msg(8, [str(1, ""), int(2, 13)]), // opset 13
]);

await mkdir(OUT, { recursive: true });
await writeFile(`${OUT}/mobilenet_v2_mid.onnx`, model);
console.log(`Wrote ${OUT}/mobilenet_v2_mid.onnx (${(model.length / 1e6).toFixed(1)} MB, ${nodes.length} nodes)`);
