import { pipeline, env } from "./vendor/transformers/transformers.js";

const vendorDir = new URL("./vendor/transformers/", import.meta.url).href;
const modelsDir = new URL("./vendor/models/", import.meta.url).href;

env.allowRemoteModels = false;
env.allowLocalModels = true;
env.useBrowserCache = false;
env.localModelPath = modelsDir;
if (env.backends?.onnx?.wasm) {
  env.backends.onnx.wasm.wasmPaths = vendorDir;
}

const fileInput = document.getElementById("file-input");
const dropzone = document.getElementById("dropzone");
const empty = document.getElementById("empty");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const slider = document.getElementById("sun-slider");
const sunValue = document.getElementById("sun-value");
const shadowSlider = document.getElementById("shadow-slider");
const shadowValue = document.getElementById("shadow-value");
const downloadBtn = document.getElementById("download-btn");
const compareBtn = document.getElementById("compare-btn");
const compareBadge = document.getElementById("compare-badge");
const processing = document.getElementById("processing");
const sunPad = document.getElementById("sun-pad");
const sunHandle = document.getElementById("sun-handle");
const sunDirLabel = document.getElementById("sun-dir-label");
const gradeBright = document.getElementById("grade-bright");
const gradeContrast = document.getElementById("grade-contrast");
const gradeHue = document.getElementById("grade-hue");
const gradeSat = document.getElementById("grade-sat");
const gradeSepia = document.getElementById("grade-sepia");
const brightValue = document.getElementById("bright-value");
const contrastValue = document.getElementById("contrast-value");
const hueValue = document.getElementById("hue-value");
const satValue = document.getElementById("sat-value");
const sepiaValue = document.getElementById("sepia-value");
const gradeSliders = [gradeBright, gradeContrast, gradeHue, gradeSat, gradeSepia];
const brushToggle = document.getElementById("brush-toggle");
const brushOverlay = document.getElementById("brush-overlay");
const brushOverlayCtx = brushOverlay.getContext("2d");
const brushSize = document.getElementById("brush-size");
const brushSizeValue = document.getElementById("brush-size-value");
const brushOpacity = document.getElementById("brush-opacity");
const brushOpacityValue = document.getElementById("brush-opacity-value");
const brushFill = document.getElementById("brush-fill");
const brushClear = document.getElementById("brush-clear");

const PREVIEW_EDGE = 960;
const EXPORT_EDGE = 2048;
const DEPTH_EDGE = 518;

let sunPos = { x: -0.7, y: -0.42 };
let source = null;
let preview = null;
let scene = null;
let lighting = null;
let lightingForce = null;
let editedPixels = null;
let showingBefore = false;
let depthEstimator = null;
let segmenter = null;
let renderTimer = 0;
let draggingSun = false;
let sunMask = null;
let brushOn = false;
let brushMode = "add";
let brushEdge = "feather";
let brushShape = "circle";
let painting = false;
let lastStamp = null;
let lineStart = null;
let cursorPos = null;
let sunlitCache = null;
let forceCache = null;
let fillCache = null;
let gradedCache = null;
let lastSunAmount = null;
let maskBeforeStroke = null;
let paintFrame = 0;

const CLS = {
  OTHER: 0,
  SKY: 1,
  WATER: 2,
  SUBJECT: 3,
  GROUND: 4,
  STRUCTURE: 5,
  FOLIAGE: 6,
};

function classFromLabel(label) {
  const l = (label || "").toLowerCase();
  if (/(sky|cloud)/.test(l)) return CLS.SKY;
  if (/(water|sea|lake|river|ocean|pond|pool|fountain|waterfall)/.test(l)) return CLS.WATER;
  if (/(person|people|animal|cat|dog|horse|cow|bird|rider)/.test(l)) return CLS.SUBJECT;
  if (/(car|bus|truck|van|bicycle|minibike|motorcycle|boat|ship|airplane)/.test(l)) return CLS.SUBJECT;
  if (/(road|sidewalk|path|earth|sand|grass|field|floor|dirt|runway|pavement|land)/.test(l)) return CLS.GROUND;
  if (/(building|house|wall|fence|skyscraper|bridge|tower|column|hovel|tent)/.test(l)) return CLS.STRUCTURE;
  if (/(tree|plant|palm|bush|forest|flower)/.test(l)) return CLS.FOLIAGE;
  return CLS.OTHER;
}

function getLight() {
  let x = sunPos.x;
  let y = sunPos.y;
  const r0 = Math.hypot(x, y);
  const r = Math.min(0.98, Math.max(0.2, r0 || 0.2));
  if (r0 > 1e-5) {
    x = (x / r0) * r;
    y = (y / r0) * r;
  } else {
    x = 0;
    y = -r;
  }
  const elev = 1 - r;
  return {
    x,
    y,
    z: 0.18 + elev * 0.95,
    screenX: x,
    screenY: y,
    height: 0.06 + elev * 0.7,
    r,
    elev,
  };
}

function describeSun(light) {
  const ang = (Math.atan2(-light.y, light.x) * 180) / Math.PI;
  const a = (ang + 360) % 360;
  let dir = "Right";
  if (a >= 45 && a < 135) dir = "Upper";
  else if (a >= 135 && a < 225) dir = "Left";
  else if (a >= 225 && a < 315) dir = "Lower";
  if (a >= 20 && a < 70) dir = "Upper right";
  else if (a >= 110 && a < 160) dir = "Upper left";
  else if (a >= 200 && a < 250) dir = "Lower left";
  else if (a >= 290 && a < 340) dir = "Lower right";
  const height = light.elev > 0.55 ? "high" : light.elev > 0.28 ? "mid" : "low";
  return `${dir} · ${height}`;
}

function updateSunHandle() {
  sunHandle.style.left = `${50 + sunPos.x * 50}%`;
  sunHandle.style.top = `${50 + sunPos.y * 50}%`;
  sunDirLabel.textContent = describeSun(getLight());
}

function setStatus(text, show = true) {
  processing.textContent = text;
  processing.hidden = !show;
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function clamp(v, a = 0, b = 1) {
  return Math.max(a, Math.min(b, v));
}

function rgbToHsl(r, g, b) {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  if (max === min) return [0, 0, l];
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  let h;
  if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
  else if (max === g) h = ((b - r) / d + 2) / 6;
  else h = ((r - g) / d + 4) / 6;
  return [h, s, l];
}

function hue2rgb(p, q, t) {
  if (t < 0) t += 1;
  if (t > 1) t -= 1;
  if (t < 1 / 6) return p + (q - p) * 6 * t;
  if (t < 1 / 2) return q;
  if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
  return p;
}

function hslToRgb(h, s, l) {
  if (s <= 1e-6) return [l, l, l];
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  return [hue2rgb(p, q, h + 1 / 3), hue2rgb(p, q, h), hue2rgb(p, q, h - 1 / 3)];
}

function signedLabel(v) {
  const n = Number(v);
  return n > 0 ? `+${n}` : `${n}`;
}

function gradeIdentity() {
  return (
    Number(gradeBright.value) === 0 &&
    Number(gradeContrast.value) === 0 &&
    Number(gradeHue.value) === 0 &&
    Number(gradeSat.value) === 0 &&
    Number(gradeSepia.value) === 0
  );
}

function applyGrade(src) {
  const out = new Uint8ClampedArray(src.length);
  if (gradeIdentity()) return new Uint8ClampedArray(src);
  const bright = Number(gradeBright.value) / 200;
  const contrast = 1 + Number(gradeContrast.value) / 100;
  const hueShift = Number(gradeHue.value) / 360;
  const satMul = 1 + Number(gradeSat.value) / 100;
  const sepiaAmt = Number(gradeSepia.value) / 100;
  for (let i = 0; i < src.length; i += 4) {
    let r = src[i] / 255 + bright;
    let g = src[i + 1] / 255 + bright;
    let b = src[i + 2] / 255 + bright;
    r = (r - 0.5) * contrast + 0.5;
    g = (g - 0.5) * contrast + 0.5;
    b = (b - 0.5) * contrast + 0.5;
    if (sepiaAmt > 0.001) {
      const sr = 0.393 * r + 0.769 * g + 0.189 * b;
      const sg = 0.349 * r + 0.686 * g + 0.168 * b;
      const sb = 0.272 * r + 0.534 * g + 0.131 * b;
      r += (sr - r) * sepiaAmt;
      g += (sg - g) * sepiaAmt;
      b += (sb - b) * sepiaAmt;
    }
    if (hueShift !== 0 || satMul !== 1) {
      const hsl = rgbToHsl(clamp(r), clamp(g), clamp(b));
      let h = hsl[0] + hueShift;
      h -= Math.floor(h);
      const s = clamp(hsl[1] * satMul);
      [r, g, b] = hslToRgb(h, s, hsl[2]);
    }
    out[i] = clamp(r) * 255;
    out[i + 1] = clamp(g) * 255;
    out[i + 2] = clamp(b) * 255;
    out[i + 3] = src[i + 3];
  }
  return out;
}

function makeNeighborFill(original, autoSun, forceSun, w, h) {
  const n = w * h;
  const wr = new Float32Array(n);
  const wg = new Float32Array(n);
  const wb = new Float32Array(n);
  const orw = new Float32Array(n);
  const ogw = new Float32Array(n);
  const obw = new Float32Array(n);
  const wt = new Float32Array(n);
  const need = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const idx = i * 4;
    const or = original[idx];
    const og = original[idx + 1];
    const ob = original[idx + 2];
    const ar = autoSun[idx];
    const ag = autoSun[idx + 1];
    const ab = autoSun[idx + 2];
    const fr = forceSun[idx];
    const fg = forceSun[idx + 1];
    const fb = forceSun[idx + 2];
    const omax = Math.max(or, og, ob);
    const omin = Math.min(or, og, ob);
    const clipped = omax > 232;
    const autoResp = (Math.abs(ar - or) + Math.abs(ag - og) + Math.abs(ab - ob)) / 3;
    const forceResp = (Math.abs(fr - or) + Math.abs(fg - og) + Math.abs(fb - ob)) / 3;
    const resp = Math.max(autoResp, forceResp);
    const warm = fr > fb + 2;
    need[i] = clipped ? 1 : autoResp < 3.5 && forceResp < 5 ? clamp((3.5 - autoResp) / 3.5) : 0;
    const donor =
      !clipped &&
      warm &&
      resp > 8 &&
      omax < 228 &&
      omax > 48 &&
      omax - omin < 64 &&
      (fr + fg + fb) / 3 > (or + og + ob) / 3 + 5;
    const wgt = donor ? clamp((resp - 6) / 16) * clamp((228 - omax) / 90) : 0;
    wr[i] = fr * wgt;
    wg[i] = fg * wgt;
    wb[i] = fb * wgt;
    orw[i] = or * wgt;
    ogw[i] = og * wgt;
    obw[i] = ob * wgt;
    wt[i] = wgt;
  }
  const rad = Math.max(12, (Math.min(w, h) / 12) | 0);
  const br = blurN(wr, w, h, rad, 2);
  const bg = blurN(wg, w, h, rad, 2);
  const bb = blurN(wb, w, h, rad, 2);
  const bor = blurN(orw, w, h, rad, 2);
  const bog = blurN(ogw, w, h, rad, 2);
  const bob = blurN(obw, w, h, rad, 2);
  const bw = blurN(wt, w, h, rad, 2);
  const out = new Uint8ClampedArray(forceSun);
  for (let i = 0; i < n; i++) {
    const idx = i * 4;
    const fillAmt = need[i];
    if (fillAmt < 0.02 || bw[i] < 1e-3) continue;
    const wsum = Math.max(bw[i], 1e-4);
    const gr = br[i] / wsum;
    const gg = bg[i] / wsum;
    const gb2 = bb[i] / wsum;
    const gor = bor[i] / wsum;
    const gog = bog[i] / wsum;
    const gob = bob[i] / wsum;
    const or = original[idx];
    const og = original[idx + 1];
    const ob = original[idx + 2];
    const fr = forceSun[idx];
    const fg = forceSun[idx + 1];
    const fb = forceSun[idx + 2];
    const omax = Math.max(or, og, ob);
    let tr;
    let tg;
    let tb;
    if (omax > 232) {
      tr = lerp(fr, gr, 0.9 * fillAmt);
      tg = lerp(fg, gg, 0.9 * fillAmt);
      tb = lerp(fb, gb2, 0.9 * fillAmt);
    } else {
      tr = lerp(fr, or + (gr - gor), fillAmt * 0.7);
      tg = lerp(fg, og + (gg - gog), fillAmt * 0.7);
      tb = lerp(fb, ob + (gb2 - gob), fillAmt * 0.7);
    }
    if (tr - tb < (fr - fb) * 0.45) {
      tr = lerp(tr, fr, 0.65);
      tg = lerp(tg, fg, 0.65);
      tb = lerp(tb, fb, 0.65);
    }
    out[idx] = clamp(tr / 255) * 255;
    out[idx + 1] = clamp(tg / 255) * 255;
    out[idx + 2] = clamp(tb / 255) * 255;
    out[idx + 3] = original[idx + 3];
  }
  return out;
}

function blendBrushLighting(original, graded, autoSun, fillSun, mask) {
  const out = new Uint8ClampedArray(original.length);
  const n = original.length / 4;
  for (let i = 0; i < n; i++) {
    const idx = i * 4;
    const b = mask ? mask[i] : 0;
    const or = original[idx];
    const og = original[idx + 1];
    const ob = original[idx + 2];
    let sr;
    let sg;
    let sb;
    if (b >= 0) {
      sr = autoSun[idx] + (fillSun[idx] - autoSun[idx]) * b;
      sg = autoSun[idx + 1] + (fillSun[idx + 1] - autoSun[idx + 1]) * b;
      sb = autoSun[idx + 2] + (fillSun[idx + 2] - autoSun[idx + 2]) * b;
    } else {
      const u = 1 + b;
      sr = or + (autoSun[idx] - or) * u;
      sg = og + (autoSun[idx + 1] - og) * u;
      sb = ob + (autoSun[idx + 2] - ob) * u;
    }
    out[idx] = clamp(graded[idx] / 255 + (sr - or) / 255) * 255;
    out[idx + 1] = clamp(graded[idx + 1] / 255 + (sg - og) / 255) * 255;
    out[idx + 2] = clamp(graded[idx + 2] / 255 + (sb - ob) / 255) * 255;
    out[idx + 3] = original[idx + 3];
  }
  return out;
}

function finishImage(src, maps, forceMaps, t, w, h, base, mask) {
  const autoSun = applyLighting(src, maps, t, w, h, base);
  const forceSun = applyLighting(src, forceMaps, t, w, h, base);
  const fillSun = makeNeighborFill(src, autoSun, forceSun, w, h);
  const graded = gradeIdentity() ? src : applyGrade(src);
  return blendBrushLighting(src, graded, autoSun, fillSun, mask);
}

function invalidatePreviewCaches() {
  sunlitCache = null;
  forceCache = null;
  fillCache = null;
  gradedCache = null;
  lastSunAmount = null;
}

function previewCaches() {
  const t = intensity();
  if (lastSunAmount !== t) {
    sunlitCache = null;
    forceCache = null;
    fillCache = null;
    lastSunAmount = t;
  }
  if (!sunlitCache) {
    sunlitCache = applyLighting(preview.pixels, lighting, t, preview.w, preview.h, scene.base);
  }
  if (!forceCache) {
    forceCache = applyLighting(preview.pixels, lightingForce, t, preview.w, preview.h, scene.base);
  }
  if (!fillCache) {
    fillCache = makeNeighborFill(preview.pixels, sunlitCache, forceCache, preview.w, preview.h);
  }
  if (!gradedCache) {
    gradedCache = gradeIdentity() ? preview.pixels : applyGrade(preview.pixels);
  }
  return { sunlit: sunlitCache, fill: fillCache, graded: gradedCache };
}

function resetSunMask(w, h, value = 0) {
  sunMask = new Float32Array(w * h);
  sunMask.fill(value);
}

function updateGradeLabels() {
  brightValue.textContent = signedLabel(gradeBright.value);
  contrastValue.textContent = signedLabel(gradeContrast.value);
  hueValue.textContent = `${signedLabel(gradeHue.value)}°`;
  satValue.textContent = signedLabel(gradeSat.value);
  sepiaValue.textContent = gradeSepia.value;
}

function resetGrade() {
  gradeBright.value = "0";
  gradeContrast.value = "0";
  gradeHue.value = "0";
  gradeSat.value = "0";
  gradeSepia.value = "0";
  updateGradeLabels();
}

function barycentric(px, py, ax, ay, bx, by, cx, cy) {
  const v0x = bx - ax;
  const v0y = by - ay;
  const v1x = cx - ax;
  const v1y = cy - ay;
  const v2x = px - ax;
  const v2y = py - ay;
  const den = v0x * v1y - v1x * v0y;
  if (Math.abs(den) < 1e-8) return -1;
  const u = (v2x * v1y - v1x * v2y) / den;
  const v = (v0x * v2y - v2x * v0y) / den;
  const t = 1 - u - v;
  return Math.min(t, u, v);
}

function stampWeight(lx, ly, radius, shape) {
  const r = Math.max(1, radius);
  if (shape === "circle") return Math.hypot(lx, ly) / r;
  if (shape === "square") return Math.max(Math.abs(lx), Math.abs(ly)) / r;
  if (shape === "rectangle") return Math.max(Math.abs(lx) / (r * 1.55), Math.abs(ly) / (r * 0.58));
  if (shape === "line") return Math.max(Math.abs(lx) / (r * 1.7), Math.abs(ly) / Math.max(2, r * 0.14));
  const m = barycentric(lx, ly, 0, -r, r * 0.95, r * 0.78, -r * 0.95, r * 0.78);
  if (m < 0) return 2;
  return 1 - m / 0.48;
}

function edgeAlpha(d, edge) {
  if (d >= 1) return 0;
  if (edge === "solid") return 1;
  if (edge === "feather") {
    const inner = 0.52;
    if (d <= inner) return 1;
    return 1 - (d - inner) / (1 - inner);
  }
  return (1 - d) * (1 - d);
}

function stampMask(cx, cy, angle) {
  if (!sunMask || !preview) return;
  const w = preview.w;
  const h = preview.h;
  const radius = Number(brushSize.value);
  const pad = radius * 1.8;
  const x0 = Math.max(0, Math.floor(cx - pad));
  const y0 = Math.max(0, Math.floor(cy - pad));
  const x1 = Math.min(w - 1, Math.ceil(cx + pad));
  const y1 = Math.min(h - 1, Math.ceil(cy + pad));
  const cos = Math.cos(-angle);
  const sin = Math.sin(-angle);
  const target = brushMode === "add" ? 1 : -1;
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const dx = x + 0.5 - cx;
      const dy = y + 0.5 - cy;
      const lx = dx * cos - dy * sin;
      const ly = dx * sin + dy * cos;
      const a =
        edgeAlpha(stampWeight(lx, ly, radius, brushShape === "line" ? "line" : brushShape), brushEdge) *
        (Number(brushOpacity.value) / 100);
      if (a <= 0) continue;
      const i = y * w + x;
      sunMask[i] += (target - sunMask[i]) * a;
    }
  }
}

function stampLine(x0, y0, x1, y1) {
  const dist = Math.hypot(x1 - x0, y1 - y0);
  const step = Math.max(2, Number(brushSize.value) * 0.22);
  const n = Math.max(1, Math.ceil(dist / step));
  const ang = Math.atan2(y1 - y0, x1 - x0);
  for (let i = 0; i <= n; i++) {
    const t = i / n;
    stampMask(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, ang);
  }
}

function canvasPoint(e) {
  const rect = canvas.getBoundingClientRect();
  if (rect.width < 1 || rect.height < 1) return null;
  const x = ((e.clientX - rect.left) / rect.width) * canvas.width;
  const y = ((e.clientY - rect.top) / rect.height) * canvas.height;
  const pad = Number(brushSize.value) * 1.2;
  if (x < -pad || y < -pad || x > canvas.width + pad || y > canvas.height + pad) return null;
  return {
    x: Math.max(0, Math.min(canvas.width - 0.01, x)),
    y: Math.max(0, Math.min(canvas.height - 0.01, y)),
  };
}

function drawBrushPreview(p, extra = null) {
  if (!preview || brushOverlay.hidden) return;
  if (brushOverlay.width !== canvas.width || brushOverlay.height !== canvas.height) {
    brushOverlay.width = canvas.width;
    brushOverlay.height = canvas.height;
  }
  const octx = brushOverlayCtx;
  octx.clearRect(0, 0, brushOverlay.width, brushOverlay.height);
  if (!brushOn || !p) return;
  const r = Number(brushSize.value);
  const color = brushMode === "add" ? "rgba(240,161,74,0.95)" : "rgba(180,210,255,0.95)";
  octx.save();
  octx.strokeStyle = color;
  octx.fillStyle = brushMode === "add" ? "rgba(240,161,74,0.12)" : "rgba(180,210,255,0.1)";
  octx.lineWidth = 1.25;
  octx.setLineDash(brushEdge === "solid" ? [] : [4, 3]);
  const drawShape = (cx, cy, ang) => {
    octx.save();
    octx.translate(cx, cy);
    octx.rotate(ang);
    octx.beginPath();
    if (brushShape === "square") octx.rect(-r, -r, r * 2, r * 2);
    else if (brushShape === "rectangle") octx.rect(-r * 1.55, -r * 0.58, r * 3.1, r * 1.16);
    else if (brushShape === "triangle") {
      octx.moveTo(0, -r);
      octx.lineTo(r * 0.95, r * 0.78);
      octx.lineTo(-r * 0.95, r * 0.78);
      octx.closePath();
    } else if (brushShape === "line") octx.rect(-r * 1.7, -Math.max(1.5, r * 0.14), r * 3.4, Math.max(3, r * 0.28));
    else octx.arc(0, 0, r, 0, Math.PI * 2);
    octx.fill();
    octx.stroke();
    octx.restore();
  };
  if (brushShape === "line" && extra) {
    octx.beginPath();
    octx.moveTo(extra.x, extra.y);
    octx.lineTo(p.x, p.y);
    octx.stroke();
    const ang = Math.atan2(p.y - extra.y, p.x - extra.x);
    drawShape(p.x, p.y, ang);
  } else {
    drawShape(p.x, p.y, 0);
  }
  octx.restore();
}

function setBrushOn(on) {
  brushOn = on && !!preview && !!lighting;
  brushToggle.setAttribute("aria-pressed", brushOn ? "true" : "false");
  dropzone.classList.toggle("brushing", brushOn);
  if (!brushOn) {
    painting = false;
    lineStart = null;
    cursorPos = null;
    if (!brushOverlay.hidden) {
      brushOverlayCtx.clearRect(0, 0, brushOverlay.width, brushOverlay.height);
    }
  }
}

function rasterize(img, maxEdge) {
  const scale = Math.min(1, maxEdge / Math.max(img.width, img.height));
  const w = Math.max(1, Math.round(img.width * scale));
  const h = Math.max(1, Math.round(img.height * scale));
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const cctx = c.getContext("2d", { willReadFrequently: true });
  cctx.drawImage(img, 0, 0, w, h);
  const pixels = new Uint8ClampedArray(cctx.getImageData(0, 0, w, h).data);
  return { canvas: c, w, h, pixels };
}

async function getSegmenter() {
  if (segmenter) return segmenter;
  setStatus("Loading scene model…");
  try {
    segmenter = await pipeline(
      "image-segmentation",
      "Xenova/segformer-b0-finetuned-ade-512-512",
      { device: "webgpu", dtype: "q8", local_files_only: true }
    );
  } catch {
    segmenter = await pipeline("image-segmentation", "Xenova/segformer-b0-finetuned-ade-512-512", {
      dtype: "q8",
      local_files_only: true,
    });
  }
  return segmenter;
}

async function getEstimator() {
  if (depthEstimator) return depthEstimator;
  setStatus("Loading depth model…");
  try {
    depthEstimator = await pipeline(
      "depth-estimation",
      "onnx-community/depth-anything-v2-small",
      { device: "webgpu", dtype: "q8", local_files_only: true }
    );
  } catch {
    depthEstimator = await pipeline(
      "depth-estimation",
      "onnx-community/depth-anything-v2-small",
      { dtype: "q8", local_files_only: true }
    );
  }
  return depthEstimator;
}

function toNearFar(data, n, nearerIsLarger) {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = data[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = max - min || 1;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let z = (data[i] - min) / range;
    if (nearerIsLarger) z = 1 - z;
    out[i] = z;
  }
  return out;
}

function upsample(src, sw, sh, dw, dh) {
  const out = new Float32Array(dw * dh);
  for (let y = 0; y < dh; y++) {
    const fy = ((y + 0.5) * sh) / dh - 0.5;
    const y0 = Math.max(0, Math.floor(fy));
    const y1 = Math.min(sh - 1, y0 + 1);
    const ty = clamp(fy - y0);
    for (let x = 0; x < dw; x++) {
      const fx = ((x + 0.5) * sw) / dw - 0.5;
      const x0 = Math.max(0, Math.floor(fx));
      const x1 = Math.min(sw - 1, x0 + 1);
      const tx = clamp(fx - x0);
      const a = src[y0 * sw + x0];
      const b = src[y0 * sw + x1];
      const c = src[y1 * sw + x0];
      const d = src[y1 * sw + x1];
      out[y * dw + x] = lerp(lerp(a, b, tx), lerp(c, d, tx), ty);
    }
  }
  return out;
}

function sampleZ(z, w, h, x, y) {
  const x0 = Math.max(0, Math.min(w - 1, x | 0));
  const y0 = Math.max(0, Math.min(h - 1, y | 0));
  const x1 = Math.min(w - 1, x0 + 1);
  const y1 = Math.min(h - 1, y0 + 1);
  const tx = clamp(x - x0);
  const ty = clamp(y - y0);
  return lerp(lerp(z[y0 * w + x0], z[y0 * w + x1], tx), lerp(z[y1 * w + x0], z[y1 * w + x1], tx), ty);
}

async function estimateDepth(img, tw, th) {
  const estimator = await getEstimator();
  const depthInput = rasterize(img, DEPTH_EDGE);
  setStatus("Estimating scene depth…");
  const blob = await new Promise((resolve) =>
    depthInput.canvas.toBlob(resolve, "image/jpeg", 0.92)
  );
  const result = await estimator(blob);
  let src;
  let sw;
  let sh;
  let nearerIsLarger = true;
  if (result.predicted_depth) {
    const t = result.predicted_depth;
    const dims = t.dims;
    sh = dims[dims.length - 2];
    sw = dims[dims.length - 1];
    src = t.data;
  } else {
    const d = result.depth;
    sw = d.width;
    sh = d.height;
    src = Float32Array.from(d.data);
    nearerIsLarger = true;
  }
  const nf = toNearFar(src, sw * sh, nearerIsLarger);
  return upsample(nf, sw, sh, tw, th);
}

function readMaskChannel(mask) {
  const data = mask.data;
  const n = mask.width * mask.height;
  if (data.length >= n * 4) {
    const out = new Uint8Array(n);
    for (let i = 0; i < n; i++) out[i] = data[i * 4];
    return out;
  }
  return data;
}

function upsampleU8(src, sw, sh, dw, dh) {
  const out = new Uint8Array(dw * dh);
  for (let y = 0; y < dh; y++) {
    const sy = Math.min(sh - 1, ((y * sh) / dh) | 0);
    for (let x = 0; x < dw; x++) {
      const sx = Math.min(sw - 1, ((x * sw) / dw) | 0);
      out[y * dw + x] = src[sy * sw + sx];
    }
  }
  return out;
}

function downsampleU8(src, sw, sh, dw, dh) {
  return upsampleU8(src, sw, sh, dw, dh);
}

async function estimateClasses(img, tw, th) {
  try {
    const seg = await getSegmenter();
    setStatus("Identifying water, sky, and subjects…");
    const input = rasterize(img, 512);
    const blob = await new Promise((resolve) => input.canvas.toBlob(resolve, "image/jpeg", 0.92));
    const output = await seg(blob);
    if (!output || !output.length) return new Uint8Array(tw * th);
    const mw = output[0].mask.width;
    const mh = output[0].mask.height;
    const best = new Float32Array(mw * mh);
    const cls = new Uint8Array(mw * mh);
    for (const item of output) {
      const id = classFromLabel(item.label);
      if (!id) continue;
      const m = readMaskChannel(item.mask);
      for (let i = 0; i < mw * mh; i++) {
        const s = m[i];
        if (s > best[i] && s > 70) {
          best[i] = s;
          cls[i] = id;
        }
      }
    }
    const full = upsampleU8(cls, mw, mh, tw, th);
    return blurClassField(full, tw, th);
  } catch (err) {
    console.warn("Segmentation failed, using depth-only lighting", err);
    return new Uint8Array(tw * th);
  }
}

function blurClassField(cls, w, h) {
  const out = new Uint8Array(cls);
  for (let y = 1; y < h - 1; y++) {
    for (let x = 1; x < w - 1; x++) {
      const counts = [0, 0, 0, 0, 0, 0, 0];
      for (let oy = -1; oy <= 1; oy++) {
        for (let ox = -1; ox <= 1; ox++) {
          counts[cls[(y + oy) * w + (x + ox)]]++;
        }
      }
      let best = 0;
      let n = 0;
      for (let c = 0; c < counts.length; c++) {
        if (counts[c] > n) {
          n = counts[c];
          best = c;
        }
      }
      out[y * w + x] = best;
    }
  }
  return out;
}

function blurN(src, w, h, radius, passes) {
  let cur = src;
  for (let i = 0; i < passes; i++) cur = blurFloat(cur, w, h, radius);
  return cur;
}

function renormalizeNormals(nx, ny, nz, n) {
  for (let i = 0; i < n; i++) {
    const len = Math.hypot(nx[i], ny[i], nz[i]) || 1;
    nx[i] /= len;
    ny[i] /= len;
    nz[i] /= len;
  }
}

function computeNormals(z, w, h) {
  const nx = new Float32Array(w * h);
  const ny = new Float32Array(w * h);
  const nz = new Float32Array(w * h);
  const tap = Math.max(3, (Math.min(w, h) / 90) | 0);
  const strength = 3.2;
  for (let y = 0; y < h; y++) {
    const y0 = Math.max(0, y - tap);
    const y1 = Math.min(h - 1, y + tap);
    for (let x = 0; x < w; x++) {
      const x0 = Math.max(0, x - tap);
      const x1 = Math.min(w - 1, x + tap);
      const dzx = (z[y * w + x1] - z[y * w + x0]) / Math.max(1, x1 - x0);
      const dzy = (z[y1 * w + x] - z[y0 * w + x]) / Math.max(1, y1 - y0);
      let vx = -dzx * strength * w;
      let vy = -dzy * strength * h;
      let vz = 1;
      const len = Math.hypot(vx, vy, vz) || 1;
      const i = y * w + x;
      nx[i] = vx / len;
      ny[i] = vy / len;
      nz[i] = vz / len;
    }
  }
  const r = Math.max(4, (Math.min(w, h) / 80) | 0);
  const bx = blurN(nx, w, h, r, 2);
  const by = blurN(ny, w, h, r, 2);
  const bz = blurN(nz, w, h, r, 2);
  renormalizeNormals(bx, by, bz, w * h);
  return { nx: bx, ny: by, nz: bz };
}

function sampleClass(cls, w, h, x, y) {
  const xx = Math.max(0, Math.min(w - 1, x | 0));
  const yy = Math.max(0, Math.min(h - 1, y | 0));
  return cls[yy * w + xx];
}

function computeVisibility(z, w, h, light, classes) {
  const vis = new Float32Array(w * h);
  const len = Math.hypot(light.screenX, light.screenY) || 1;
  const lx = light.screenX / len;
  const ly = light.screenY / len;
  const steps = 64;
  const step = Math.max(1.15, Math.min(w, h) / 420);
  const slope = light.height;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const recv = classes ? classes[y * w + x] : CLS.OTHER;
      if (recv === CLS.SKY) {
        vis[y * w + x] = 1;
        continue;
      }
      const z0 = z[y * w + x];
      let v = 1;
      const maxGap = recv === CLS.WATER ? 0.045 : 0.14;
      for (let s = 1; s <= steps; s++) {
        const px = x + lx * s * step;
        const py = y + ly * s * step;
        if (px < 1 || py < 1 || px >= w - 1 || py >= h - 1) break;
        const zs = sampleZ(z, w, h, px, py);
        const dist = (s * step) / Math.max(w, h);
        const gap = z0 - zs;
        if (gap <= 0.022 + dist * slope) continue;
        if (gap > maxGap) continue;
        const oc = classes ? sampleClass(classes, w, h, px, py) : CLS.OTHER;
        if (recv === CLS.WATER && oc !== CLS.STRUCTURE && oc !== CLS.GROUND) continue;
        if (oc === CLS.SUBJECT && recv !== CLS.GROUND && recv !== CLS.SUBJECT) continue;
        if (oc === CLS.SKY) continue;
        const occl = clamp((gap - dist * slope) / 0.11);
        const fade = 1 - s / (steps + 4);
        v = Math.min(v, 1 - 0.92 * occl * (0.7 + 0.3 * fade));
        if (v < 0.08) break;
      }
      vis[y * w + x] = v;
    }
  }
  return blurN(vis, w, h, Math.max(3, (Math.min(w, h) / 140) | 0), 2);
}

function computeGloss(pixels, w, h) {
  const gloss = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = (y * w + x) * 4;
      const r = pixels[idx] / 255;
      const g = pixels[idx + 1] / 255;
      const b = pixels[idx + 2] / 255;
      const maxc = Math.max(r, g, b);
      const minc = Math.min(r, g, b);
      const sat = maxc < 1e-4 ? 0 : 1 - minc / maxc;
      gloss[y * w + x] = (1 - sat) ** 2 * clamp((maxc - 0.35) / 0.45);
    }
  }
  return blurN(gloss, w, h, Math.max(5, (Math.min(w, h) / 70) | 0), 2);
}

function blurFloat(src, w, h, radius) {
  const tmp = new Float32Array(src.length);
  const out = new Float32Array(src.length);
  const r = radius;
  const div = r * 2 + 1;
  for (let y = 0; y < h; y++) {
    let acc = 0;
    for (let k = -r; k <= r; k++) acc += src[y * w + Math.min(w - 1, Math.max(0, k))];
    for (let x = 0; x < w; x++) {
      tmp[y * w + x] = acc / div;
      acc += src[y * w + Math.min(w - 1, Math.max(0, x + r + 1))] - src[y * w + Math.min(w - 1, Math.max(0, x - r))];
    }
  }
  for (let x = 0; x < w; x++) {
    let acc = 0;
    for (let k = -r; k <= r; k++) acc += tmp[Math.min(h - 1, Math.max(0, k)) * w + x];
    for (let y = 0; y < h; y++) {
      out[y * w + x] = acc / div;
      acc += tmp[Math.min(h - 1, Math.max(0, y + r + 1)) * w + x] - tmp[Math.min(h - 1, Math.max(0, y - r)) * w + x];
    }
  }
  return out;
}

function downsample(src, sw, sh, dw, dh) {
  const out = new Float32Array(dw * dh);
  for (let y = 0; y < dh; y++) {
    const sy = Math.min(sh - 1, ((y * sh) / dh) | 0);
    for (let x = 0; x < dw; x++) {
      const sx = Math.min(sw - 1, ((x * sw) / dw) | 0);
      out[y * dw + x] = src[sy * sw + sx];
    }
  }
  return out;
}

function computeVisibilityPreview(z, w, h, light, classes) {
  const sw = Math.max(120, (w / 2) | 0);
  const sh = Math.max(120, (h / 2) | 0);
  const zs = downsample(z, w, h, sw, sh);
  const cs = classes ? downsampleU8(classes, w, h, sw, sh) : null;
  const visS = computeVisibility(zs, sw, sh, light, cs);
  return upsample(visS, sw, sh, w, h);
}

function shadowAmount() {
  return Number(shadowSlider.value) / 100;
}

function buildLighting(pixels, scn, light, preview = true, remarch = true, forceLit = false) {
  const { z, w, h, nx, ny, nz, gloss, cls } = scn;
  if (remarch || !scn.vis) {
    scn.vis = preview
      ? computeVisibilityPreview(z, w, h, light, cls)
      : computeVisibility(z, w, h, light, cls);
  }
  const vis = scn.vis;
  const shadow = shadowAmount();
  const llen = Math.hypot(light.x, light.y, light.z) || 1;
  const lx = light.x / llen;
  const ly = light.y / llen;
  const lz = light.z / llen;
  const hx = lx;
  const hy = ly;
  const hz = lz + 1;
  const hlen = Math.hypot(hx, hy, hz);

  const gainR = new Float32Array(w * h);
  const gainG = new Float32Array(w * h);
  const gainB = new Float32Array(w * h);
  const addR = new Float32Array(w * h);
  const addG = new Float32Array(w * h);
  const addB = new Float32Array(w * h);
  const bloom = new Float32Array(w * h);

  const litMap = new Float32Array(w * h);
  const specMap = new Float32Array(w * h);
  const farMap = new Float32Array(w * h);

  for (let i = 0; i < w * h; i++) {
    const force = forceLit ? 1 : 0;
    const ndotl = clamp(nx[i] * lx + ny[i] * ly + nz[i] * lz);
    const ndoth = clamp((nx[i] * hx + ny[i] * hy + nz[i] * hz) / hlen);
    const kind = cls ? cls[i] : CLS.OTHER;
    const v =
      force || kind === CLS.SKY || kind === CLS.WATER ? 1 : lerp(1, vis[i], shadow);
    const lit = v * ndotl;
    litMap[i] = force ? lerp(lit, 0.22 + 0.78 * ndotl, 1) : lit;
    specMap[i] = v * Math.pow(ndoth, lerp(32, 70, gloss[i])) * (0.02 + 0.4 * gloss[i]);
    farMap[i] = z[i];
  }

  const smoothR = Math.max(5, (Math.min(w, h) / 90) | 0);
  const lit = blurN(litMap, w, h, smoothR, 2);
  const spec = blurN(specMap, w, h, Math.max(2, (smoothR / 2) | 0), 1);

  for (let i = 0; i < w * h; i++) {
    const ndotl = clamp(nx[i] * lx + ny[i] * ly + nz[i] * lz);
    const ndoth = clamp((nx[i] * hx + ny[i] * hy + nz[i] * hz) / hlen);
    const kind = cls ? cls[i] : CLS.OTHER;
    const x = i % w;
    const y = (i / w) | 0;
    const sx = x / w;
    const sy = y / h;
    const sunU = 0.5 + light.x * 0.48;
    const sunV = 0.5 + light.y * 0.48;
    const towardSun = clamp(1 - Math.hypot(sx - sunU, sy - sunV) * 1.35);
    const sunGlow = towardSun * towardSun;

    let L = clamp(lit[i]);
    let Sp = spec[i];
    let warm = 0;
    let cool = 0;
    let lift = 0;
    const pr = pixels[i * 4] / 255;
    const pg = pixels[i * 4 + 1] / 255;
    const pb = pixels[i * 4 + 2] / 255;
    const lum = 0.2126 * pr + 0.7152 * pg + 0.0722 * pb;
    const chroma = Math.max(pr, pg, pb) - Math.min(pr, pg, pb);
    const skyish =
      kind === CLS.SKY ||
      (kind === CLS.OTHER &&
        sy < 0.42 &&
        lum > 0.58 &&
        pb > pg * 1.06 &&
        pb > pr * 1.04 &&
        chroma > 0.06);

    if (skyish) {
      warm = sunGlow * 0.55;
      cool = (1 - sunGlow) * 0.1;
      lift = sunGlow * 0.1;
      Sp = sunGlow * 0.08;
    } else if (kind === CLS.WATER) {
      warm = 0.18 + 0.45 * L;
      const u = sx - 0.5 - light.x * 0.28;
      Sp = 0.06 + 0.5 * Math.pow(ndoth, 22) + Math.exp(-u * u * 42) * 0.32;
      lift = 0.04;
    } else if (kind === CLS.SUBJECT) {
      warm = 0.14 + 0.92 * Math.pow(L, 1.05);
      cool = 0.28 * (1 - L);
      Sp = spec[i] * 0.8 + (1 - ndotl) * L * 0.07;
      lift = L * 0.08;
    } else if (kind === CLS.FOLIAGE) {
      warm = 0.12 + 0.55 * L;
      cool = 0.14 * (1 - L);
      Sp *= 0.4;
    } else {
      warm = 0.1 + 0.78 * Math.pow(L, 1.08);
      cool = 0.2 * (1 - L);
      Sp *= 0.65;
      if (kind === CLS.STRUCTURE) warm += L * 0.12;
    }

    gainR[i] = 1 + warm * 0.72 - cool * 0.1 + lift;
    gainG[i] = 1 + warm * 0.22 - cool * 0.03 + lift * 0.5;
    gainB[i] = 1 - warm * 0.38 + cool * 0.14 + lift * 0.1;

    addR[i] = Sp * 0.85 + warm * 0.04;
    addG[i] = Sp * 0.55 + warm * 0.018;
    addB[i] = Sp * 0.16;
    bloom[i] = (skyish ? sunGlow * 0.14 : 0) + Sp * 0.5 + (kind === CLS.SUBJECT ? L * L * 0.08 : L * L * 0.03);
  }

  const glow = blurN(bloom, w, h, Math.max(5, (Math.min(w, h) / 60) | 0), 1);
  for (let i = 0; i < w * h; i++) {
    addR[i] += glow[i] * 0.45;
    addG[i] += glow[i] * 0.28;
    addB[i] += glow[i] * 0.08;
  }

  return { gainR, gainG, gainB, addR, addG, addB };
}

function intensity() {
  return Number(slider.value) / 100;
}

function blurImage(pixels, w, h, radius) {
  const n = w * h;
  const r = new Float32Array(n);
  const g = new Float32Array(n);
  const b = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    r[i] = pixels[i * 4] / 255;
    g[i] = pixels[i * 4 + 1] / 255;
    b[i] = pixels[i * 4 + 2] / 255;
  }
  return {
    r: blurN(r, w, h, radius, 1),
    g: blurN(g, w, h, radius, 1),
    b: blurN(b, w, h, radius, 1),
  };
}

function compressPeak(r, g, b) {
  const peak = Math.max(r, g, b, 1e-6);
  const shoulder = 0.72;
  if (peak <= shoulder) return [r, g, b];
  const over = peak - shoulder;
  const rolled = shoulder + over / (1 + over * 2.6);
  const s = Math.min(1, rolled / peak);
  return [r * s, g * s, b * s];
}

function applyLighting(src, maps, t, w, h, base) {
  const out = new Uint8ClampedArray(src.length);
  if (t <= 0.001) return new Uint8ClampedArray(src);
  const k = t;
  const { gainR, gainG, gainB, addR, addG, addB } = maps;
  for (let i = 0; i < w * h; i++) {
    const idx = i * 4;
    const r = src[idx] / 255;
    const g = src[idx + 1] / 255;
    const b = src[idx + 2] / 255;
    const br = base ? base.r[i] : r;
    const bg = base ? base.g[i] : g;
    const bb = base ? base.b[i] : b;
    const gR = lerp(1, gainR[i], k);
    const gG = lerp(1, gainG[i], k);
    const gB = lerp(1, gainB[i], k);
    let rr = br * gR + addR[i] * k;
    let gg = bg * gG + addG[i] * k;
    let bb2 = bb * gB + addB[i] * k;
    [rr, gg, bb2] = compressPeak(rr, gg, bb2);
    const clip = clamp((Math.max(r, g, b) - 0.78) / 0.2);
    const detail = 1 - clip;
    rr += (r - br) * detail * gR;
    gg += (g - bg) * detail * gG;
    bb2 += (b - bb) * detail * gB;
    out[idx] = clamp(rr) * 255;
    out[idx + 1] = clamp(gg) * 255;
    out[idx + 2] = clamp(bb2) * 255;
    out[idx + 3] = src[idx + 3];
  }
  return out;
}

function showPixels(pixels, w, h) {
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  ctx.putImageData(new ImageData(pixels, w, h), 0, 0);
  if (brushOverlay.width !== w || brushOverlay.height !== h) {
    brushOverlay.width = w;
    brushOverlay.height = h;
  }
  drawBrushPreview(cursorPos, lineStart);
}

function setShowingBefore(on) {
  showingBefore = on;
  compareBtn.setAttribute("aria-pressed", on ? "true" : "false");
  compareBadge.hidden = !preview;
  compareBadge.textContent = on ? "Before" : "After";
  if (!preview) return;
  if (on) showPixels(preview.pixels, preview.w, preview.h);
  else if (editedPixels) showPixels(editedPixels, preview.w, preview.h);
}

function paintPreview() {
  if (!preview || !lighting || !lightingForce) return;
  const { sunlit, fill, graded } = previewCaches();
  editedPixels = blendBrushLighting(preview.pixels, graded, sunlit, fill, sunMask);
  if (!showingBefore) showPixels(editedPixels, preview.w, preview.h);
  compareBadge.hidden = false;
  compareBadge.textContent = showingBefore ? "Before" : "After";
}

function requestPaintPreview() {
  if (paintFrame) return;
  paintFrame = requestAnimationFrame(() => {
    paintFrame = 0;
    paintPreview();
  });
}

function render(immediate) {
  if (!preview || !lighting) return;
  sunValue.textContent = `${slider.value}%`;
  shadowValue.textContent = `${shadowSlider.value}%`;
  clearTimeout(renderTimer);
  if (immediate || painting) {
    paintPreview();
    return;
  }
  renderTimer = setTimeout(paintPreview, 16);
}

function makeScene(pixels, zRaw, w, h, classes) {
  const z = blurN(zRaw, w, h, Math.max(3, (Math.min(w, h) / 110) | 0), 3);
  const normals = computeNormals(z, w, h);
  return {
    z,
    w,
    h,
    nx: normals.nx,
    ny: normals.ny,
    nz: normals.nz,
    gloss: computeGloss(pixels, w, h),
    base: blurImage(pixels, w, h, 2),
    cls: classes || new Uint8Array(w * h),
    vis: null,
  };
}

function setControlsEnabled(on) {
  slider.disabled = !on;
  shadowSlider.disabled = !on;
  gradeSliders.forEach((el) => {
    el.disabled = !on;
  });
  brushToggle.disabled = !on;
  brushSize.disabled = !on;
  brushOpacity.disabled = !on;
  brushFill.disabled = !on;
  brushClear.disabled = !on;
  document.querySelectorAll(".brush-panel .seg-btn").forEach((el) => {
    el.disabled = !on;
  });
  downloadBtn.disabled = !on;
  compareBtn.disabled = !on;
  sunPad.setAttribute("aria-disabled", on ? "false" : "true");
  if (!on) setBrushOn(false);
}

function rebuildLighting(remarch = true) {
  if (!preview || !scene) return;
  lighting = buildLighting(preview.pixels, scene, getLight(), true, remarch, false);
  lightingForce = buildLighting(preview.pixels, scene, getLight(), true, false, true);
  invalidatePreviewCaches();
  render(true);
  setStatus("", false);
}

let lightingQueued = false;
let remarchQueued = false;

function queueRebuild(remarch = true) {
  remarchQueued = remarchQueued || remarch;
  if (lightingQueued) return;
  lightingQueued = true;
  requestAnimationFrame(() => {
    lightingQueued = false;
    const doMarch = remarchQueued;
    remarchQueued = false;
    rebuildLighting(doMarch);
  });
}

async function loadImage(file) {
  if (!file || !file.type.startsWith("image/")) return;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = async () => {
    URL.revokeObjectURL(url);
    source = img;
    preview = rasterize(img, PREVIEW_EDGE);
    empty.hidden = true;
    canvas.hidden = false;
    resetGrade();
    resetSunMask(preview.w, preview.h, 0);
    invalidatePreviewCaches();
    brushOverlay.hidden = false;
    showPixels(new Uint8ClampedArray(preview.pixels), preview.w, preview.h);
    editedPixels = null;
    setShowingBefore(false);
    compareBadge.hidden = true;
    setControlsEnabled(false);
    try {
      const z = await estimateDepth(img, preview.w, preview.h);
      const classes = await estimateClasses(img, preview.w, preview.h);
      scene = makeScene(preview.pixels, z, preview.w, preview.h, classes);
      rebuildLighting();
      setControlsEnabled(true);
      if (Number(slider.value) === 0) slider.value = "100";
      render(true);
      setStatus("", false);
    } catch (err) {
      console.error(err);
      setStatus("Could not estimate depth. Check your network and try again.", true);
    }
  };
  img.src = url;
}

fileInput.addEventListener("change", (e) => loadImage(e.target.files[0]));
compareBtn.addEventListener("click", () => {
  if (!preview || !editedPixels) return;
  setShowingBefore(!showingBefore);
});
slider.addEventListener("input", () => {
  sunValue.textContent = `${slider.value}%`;
  invalidatePreviewCaches();
  render(false);
});
shadowSlider.addEventListener("input", () => {
  shadowValue.textContent = `${shadowSlider.value}%`;
  queueRebuild(false);
});
gradeSliders.forEach((el) => {
  el.addEventListener("input", () => {
    updateGradeLabels();
    gradedCache = null;
    render(false);
  });
});

function setSeg(group, attr, value) {
  group.querySelectorAll(".seg-btn").forEach((btn) => {
    btn.setAttribute("aria-pressed", btn.dataset[attr] === value ? "true" : "false");
  });
}

document.getElementById("brush-mode").addEventListener("click", (e) => {
  const btn = e.target.closest("[data-mode]");
  if (!btn || btn.disabled) return;
  brushMode = btn.dataset.mode;
  setSeg(e.currentTarget, "mode", brushMode);
  drawBrushPreview(cursorPos, lineStart);
});

document.getElementById("brush-edge").addEventListener("click", (e) => {
  const btn = e.target.closest("[data-edge]");
  if (!btn || btn.disabled) return;
  brushEdge = btn.dataset.edge;
  setSeg(e.currentTarget, "edge", brushEdge);
  drawBrushPreview(cursorPos, lineStart);
});

document.getElementById("brush-shape").addEventListener("click", (e) => {
  const btn = e.target.closest("[data-shape]");
  if (!btn || btn.disabled) return;
  brushShape = btn.dataset.shape;
  setSeg(e.currentTarget, "shape", brushShape);
  drawBrushPreview(cursorPos, lineStart);
});

brushToggle.addEventListener("click", () => {
  if (brushToggle.disabled) return;
  setBrushOn(!brushOn);
  drawBrushPreview(cursorPos, lineStart);
});

brushSize.addEventListener("input", () => {
  brushSizeValue.textContent = brushSize.value;
  drawBrushPreview(cursorPos, lineStart);
});
brushOpacity.addEventListener("input", () => {
  brushOpacityValue.textContent = `${brushOpacity.value}%`;
});

brushFill.addEventListener("click", () => {
  if (!sunMask) return;
  sunMask.fill(1);
  render(true);
});

brushClear.addEventListener("click", () => {
  if (!sunMask) return;
  sunMask.fill(-1);
  render(true);
});

const viewStack = canvas.parentElement;

viewStack.addEventListener("pointerdown", (e) => {
  if (!brushOn || !preview || !lighting) return;
  e.preventDefault();
  e.stopPropagation();
  viewStack.setPointerCapture(e.pointerId);
  const p = canvasPoint(e);
  if (!p) return;
  painting = true;
  showingBefore = false;
  compareBtn.setAttribute("aria-pressed", "false");
  cursorPos = p;
  if (brushShape === "line") {
    lineStart = p;
    maskBeforeStroke = new Float32Array(sunMask);
    drawBrushPreview(p, lineStart);
    paintPreview();
    return;
  }
  lastStamp = p;
  stampMask(p.x, p.y, 0);
  paintPreview();
});

viewStack.addEventListener("pointermove", (e) => {
  const p = canvasPoint(e);
  cursorPos = p;
  if (!brushOn) return;
  if (!p) return;
  if (!painting) {
    drawBrushPreview(p);
    return;
  }
  if (brushShape === "line") {
    if (maskBeforeStroke) sunMask.set(maskBeforeStroke);
    stampLine(lineStart.x, lineStart.y, p.x, p.y);
    requestPaintPreview();
    drawBrushPreview(p, lineStart);
    return;
  }
  const prev = lastStamp || p;
  stampLine(prev.x, prev.y, p.x, p.y);
  lastStamp = p;
  requestPaintPreview();
  drawBrushPreview(p);
});

viewStack.addEventListener("pointerup", (e) => {
  if (!painting) return;
  const p = canvasPoint(e) || lastStamp;
  if (brushShape === "line" && lineStart && p) {
    if (maskBeforeStroke) sunMask.set(maskBeforeStroke);
    stampLine(lineStart.x, lineStart.y, p.x, p.y);
  }
  painting = false;
  lastStamp = null;
  lineStart = null;
  maskBeforeStroke = null;
  paintPreview();
});

viewStack.addEventListener("pointerleave", () => {
  if (painting) return;
  cursorPos = null;
  drawBrushPreview(null);
});

viewStack.addEventListener("lostpointercapture", () => {
  if (!painting) return;
  painting = false;
  lastStamp = null;
  lineStart = null;
  maskBeforeStroke = null;
  paintPreview();
});

window.addEventListener("keydown", (e) => {
  if (!preview || brushSize.disabled) return;
  if (e.target.matches("input, textarea")) return;
  if (e.key === "[") {
    brushSize.value = String(Math.max(8, Number(brushSize.value) - 8));
    brushSize.dispatchEvent(new Event("input"));
  } else if (e.key === "]") {
    brushSize.value = String(Math.min(280, Number(brushSize.value) + 8));
    brushSize.dispatchEvent(new Event("input"));
  } else if (e.key === "b" || e.key === "B") {
    if (!brushToggle.disabled) setBrushOn(!brushOn);
  }
});

downloadBtn.addEventListener("click", async () => {
  if (!source || !scene) return;
  setStatus("Exporting…", true);
  await new Promise((r) => requestAnimationFrame(r));
  const full = rasterize(source, EXPORT_EDGE);
  const z = upsample(scene.z, scene.w, scene.h, full.w, full.h);
  const cls = upsampleU8(scene.cls, scene.w, scene.h, full.w, full.h);
  const exportScene = makeScene(full.pixels, z, full.w, full.h, cls);
  const maps = buildLighting(full.pixels, exportScene, getLight(), false, true, false);
  const forceMaps = buildLighting(full.pixels, exportScene, getLight(), false, false, true);
  const exportMask = sunMask ? upsample(sunMask, preview.w, preview.h, full.w, full.h) : null;
  const pixels = finishImage(
    full.pixels,
    maps,
    forceMaps,
    intensity(),
    full.w,
    full.h,
    exportScene.base,
    exportMask
  );
  const exportCanvas = document.createElement("canvas");
  exportCanvas.width = full.w;
  exportCanvas.height = full.h;
  exportCanvas.getContext("2d").putImageData(new ImageData(pixels, full.w, full.h), 0, 0);
  const a = document.createElement("a");
  a.download = "golden-hour.png";
  a.href = exportCanvas.toDataURL("image/png");
  a.click();
  setStatus("", false);
});

["dragenter", "dragover"].forEach((ev) => {
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
});
["dragleave", "drop"].forEach((ev) => {
  dropzone.addEventListener(ev, (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  });
});
dropzone.addEventListener("drop", (e) => {
  loadImage(e.dataTransfer.files[0]);
});

function setSunFromEvent(e) {
  const rect = sunPad.getBoundingClientRect();
  let x = (e.clientX - (rect.left + rect.width / 2)) / (rect.width / 2);
  let y = (e.clientY - (rect.top + rect.height / 2)) / (rect.height / 2);
  const r = Math.hypot(x, y);
  if (r > 1) {
    x /= r;
    y /= r;
  }
  sunPos = { x, y };
  updateSunHandle();
  queueRebuild();
}

sunPad.addEventListener("pointerdown", (e) => {
  if (sunPad.getAttribute("aria-disabled") === "true") return;
  draggingSun = true;
  sunPad.setPointerCapture(e.pointerId);
  setSunFromEvent(e);
});
sunPad.addEventListener("pointermove", (e) => {
  if (!draggingSun) return;
  setSunFromEvent(e);
});
sunPad.addEventListener("pointerup", () => {
  draggingSun = false;
});
sunPad.addEventListener("pointercancel", () => {
  draggingSun = false;
});

updateSunHandle();
updateGradeLabels();
