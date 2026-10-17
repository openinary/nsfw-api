// Build-time step for the nudity model: converts nsfwjs's MobileNetV2 (a Keras
// model stored in TensorFlow.js format) into an ONNX file, so the server runs
// both models on ONNX Runtime and TensorFlow.js never ships in the image.
//
// The weights keep nsfwjs's uint8 quantisation (2.6 MB instead of 10.5 MB);
// the graph dequantises them in float64 exactly like TensorFlow.js does
// (q * scale + min, rounded once to float32), and ONNX Runtime folds that into
// plain constants when the model loads.
// Output: models/nsfw/mobilenet_v2.onnx
import { mkdir, writeFile } from "node:fs/promises";
import { MobileNetV2Model } from "nsfwjs/models/mobilenet_v2";

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
const ATTR = { FLOAT: 1, INT: 2, INTS: 7 };

const tensor = (name: string, dataType: number, dims: number[], raw: Buffer) =>
  Buffer.concat([...dims.map((d) => int(1, d)), int(2, dataType), str(8, name), bytes(9, raw)]);

type AttrValue = number | number[];

function attribute(name: string, value: AttrValue) {
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
const modelJson = await MobileNetV2Model.modelJson();
const { modelTopology, weightsManifest } = modelJson.default ?? modelJson;
const shards = await Promise.all(MobileNetV2Model.weightBundles.map((load) => load()));
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

function activation(x: string, fn: string) {
  if (fn === "linear") return x;
  if (fn === "relu") return node("Relu", [x]);
  if (fn === "softmax") return node("Softmax", [x], { axis: -1 });
  throw new Error(`Unsupported activation ${fn}`);
}

// Keras "same" padding is only symmetric for stride 1 and odd kernels, the only case MobileNetV2 uses.
function convPads(cfg: LayerConfig, extra?: number[]) {
  const [kh, kw] = cfg.kernel_size;
  let pads = [0, 0, 0, 0]; // top, left, bottom, right
  if (cfg.padding === "same") {
    if (cfg.strides.some((s) => s !== 1) || kh % 2 === 0 || kw % 2 === 0) {
      throw new Error(`Unsupported "same" padding in ${cfg.name}`);
    }
    pads = [(kh - 1) / 2, (kw - 1) / 2, (kh - 1) / 2, (kw - 1) / 2];
  }
  if (extra) pads = pads.map((p, i) => p + extra[i]);
  return [pads[0], pads[1], pads[2], pads[3]];
}

// ---------------------------------------------------------------------------
// Keras layers -> ONNX (NCHW inside the graph)

// The part of the Keras topology this converter reads (TensorFlow.js types it as {}).
interface LayerConfig {
  name: string;
  kernel_size: number[];
  strides: number[];
  padding: "same" | "valid" | number[][];
  dilation_rate: number[];
  use_bias: boolean;
  activation: string;
  center: boolean;
  scale: boolean;
  axis: number;
  epsilon: number;
  negative_slope?: number;
  threshold?: number;
  max_value?: number | null;
  pool_size: number[];
}
interface KerasTopology {
  model_config: {
    config: {
      input_layers: [string, number, number][];
      output_layers: [string, number, number][];
      layers: { name: string; class_name: string; config: LayerConfig; inbound_nodes: [string, ...unknown[]][][] }[];
    };
  };
}
const config = (modelTopology as KerasTopology).model_config.config;
const tensors = new Map<string, string>(); // layer name -> ONNX tensor name
const pendingPads = new Map<string, number[]>(); // ZeroPadding2D output -> pads, folded into the consuming conv

const inputName = config.input_layers[0][0];
tensors.set(inputName, node("Transpose", ["input"], { perm: [0, 3, 1, 2] }, "input_nchw"));

for (const layer of config.layers) {
  const { class_name: type, config: cfg, inbound_nodes: inbound } = layer;
  if (type === "InputLayer") continue;
  const inputs = inbound[0].map(([name]) => name);
  const x = tensors.get(inputs[0])!;
  let y: string;
  switch (type) {
    case "ZeroPadding2D": {
      const [[t, b], [l, r]] = cfg.padding as number[][];
      pendingPads.set(layer.name, [t, l, b, r]);
      y = x;
      break;
    }
    case "Conv2D":
    case "DepthwiseConv2D": {
      if (cfg.dilation_rate.some((d) => d !== 1)) throw new Error(`Unsupported dilation in ${layer.name}`);
      const depthwise = type === "DepthwiseConv2D";
      const k = weights.get(`${layer.name}/${depthwise ? "depthwise_kernel" : "kernel"}`)!;
      const [kh, kw, cin, cout] = k.shape;
      // Keras kernel [kh, kw, in, out|multiplier] -> ONNX [out, in/group, kh, kw]
      const w = weight(`${layer.name}/${depthwise ? "depthwise_kernel" : "kernel"}`, {
        perm: depthwise ? [2, 3, 0, 1] : [3, 2, 0, 1],
        reshape: depthwise ? [cin * cout, 1, kh, kw] : undefined,
      });
      const ins = [x, w];
      if (cfg.use_bias) ins.push(weight(`${layer.name}/bias`));
      y = node("Conv", ins, {
        kernel_shape: [kh, kw],
        strides: cfg.strides,
        pads: convPads(cfg, pendingPads.get(inputs[0])),
        group: depthwise ? cin : 1,
      });
      y = activation(y, cfg.activation);
      break;
    }
    case "BatchNormalization": {
      if (!cfg.center || !cfg.scale || ![-1, 3].includes(cfg.axis)) throw new Error(`Unsupported BN ${layer.name}`);
      y = node(
        "BatchNormalization",
        [x, ...["gamma", "beta", "moving_mean", "moving_variance"].map((p) => weight(`${layer.name}/${p}`))],
        { epsilon: cfg.epsilon },
      );
      break;
    }
    case "ReLU": {
      if (cfg.negative_slope || cfg.threshold) throw new Error(`Unsupported ReLU options in ${layer.name}`);
      y =
        cfg.max_value == null
          ? node("Relu", [x])
          : node("Clip", [x, f32(`${layer.name}_lo`, 0), f32(`${layer.name}_hi`, cfg.max_value)]);
      break;
    }
    case "Add":
      y = inputs.map((n) => tensors.get(n)!).reduce((a, b) => node("Add", [a, b]));
      break;
    case "AveragePooling2D":
      if (cfg.padding !== "valid") throw new Error(`Unsupported pooling padding in ${layer.name}`);
      y = node("AveragePool", [x], { kernel_shape: cfg.pool_size, strides: cfg.strides });
      break;
    case "Flatten":
      // The feature map is 1x1 here, so NCHW and NHWC flatten to the same order.
      y = node("Flatten", [x], { axis: 1 });
      break;
    case "Dense":
      y = node("Gemm", [x, weight(`${layer.name}/kernel`), ...(cfg.use_bias ? [weight(`${layer.name}/bias`)] : [])]);
      y = activation(y, cfg.activation);
      break;
    case "Dropout":
      y = x; // inference: identity
      break;
    default:
      throw new Error(`Unsupported layer ${type} (${layer.name})`);
  }
  tensors.set(layer.name, y);
}
if (weights.size) throw new Error(`Unused weights: ${[...weights.keys()].join(", ")}`);

const output = node("Identity", [tensors.get(config.output_layers[0][0])!], {}, "probabilities");

const graph = Buffer.concat([
  ...nodes.map((n) => msg(1, [n])),
  str(2, "nsfwjs_mobilenet_v2"),
  ...initializers.map((t) => msg(5, [t])),
  msg(11, [valueInfo("input", [1, SIZE, SIZE, 3])]),
  msg(12, [valueInfo(output, [1, 5])]),
]);
const model = Buffer.concat([
  int(1, 8), // IR version
  str(2, "openinary/nsfw-api scripts/prepare-nsfw.ts"),
  str(6, "nsfwjs 4.4.0 MobileNetV2 (MIT, github.com/infinitered/nsfwjs), converted to ONNX"),
  msg(7, [graph]),
  msg(8, [str(1, ""), int(2, 13)]), // opset 13
]);

await mkdir(OUT, { recursive: true });
await writeFile(`${OUT}/mobilenet_v2.onnx`, model);
console.log(`Wrote ${OUT}/mobilenet_v2.onnx (${(model.length / 1e6).toFixed(1)} MB, ${nodes.length} nodes)`);
