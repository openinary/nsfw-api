// Zero-shot categories for the CLIP module. Each category is described by a few
// text prompts; an image's score for a category is the share of probability
// CLIP gives to that category's prompts, against every enabled category and the
// OTHER prompts below. Editing this file requires a rebuild (prompts are turned
// into embeddings at build time by scripts/prepare-clip.ts).
export const CATEGORIES: Record<string, string[]> = {
  weapon: [
    "a photo of a gun",
    "a photo of a handgun",
    "a photo of a rifle",
    "a photo of an assault rifle",
    "a person holding a gun",
    "a person pointing a gun at the camera",
    "a person threatening someone with a knife",
  ],
  gore: [
    "a photo of blood and gore",
    "a photo of a bloody wound",
    "a photo of a dead body",
    "a photo of a mutilated body",
    "a graphic photo of a violent injury",
  ],
  drugs: [
    "a photo of illegal drugs",
    "a photo of lines of cocaine",
    "a photo of a syringe used to inject drugs",
    "a photo of a bag of white powder and pills",
    "a person smoking a joint",
  ],
};

// Everything else, never flagged by CLIP. It must cover what images commonly
// show, or CLIP pushes their probability onto the closest category: without the
// "people and bodies" prompts, suggestive photos scored up to 39% gore and 27%
// drugs ("a dead body", "a person smoking a joint" were their nearest match).
// Nudity itself is nsfwjs's job, so it belongs here too.
export const OTHER: string[] = [
  "a photo",
  "a photo of people",
  "a selfie",
  // people and bodies
  "a portrait of a person",
  "a fashion photo of a model posing",
  "a photo of a shirtless man",
  "a photo of a woman in lingerie",
  "a photo of a person lying on a bed",
  "a nude photo",
  "an artistic nude",
  "a close-up of skin",
  "a photo of a person at the beach",
  "a group photo of friends",
  "a photo of a landscape",
  "a photo of a city street",
  "a photo of a building",
  "a photo of a room",
  "a photo of food",
  "a photo of a meal with ketchup",
  "a photo of an animal",
  "a photo of a pet",
  "a photo of a car",
  "a photo of everyday objects",
  "a photo of tools",
  "a photo of a power drill",
  "a photo of a kitchen knife on a cutting board",
  "a photo of a toy water gun",
  "a photo of a video game",
  "a photo of red paint",
  "a photo of medicine at a pharmacy",
  "a photo of a doctor with a patient",
  "a photo of sports",
  "a screenshot",
  "a drawing",
  "a cartoon",
  "a document with text",
  "a logo",
];
