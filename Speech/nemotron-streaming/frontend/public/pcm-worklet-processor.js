// AudioWorklet processor, served as a plain static file (public/) and loaded
// via audioWorklet.addModule('/pcm-worklet-processor.js') — this runs in the
// isolated AudioWorkletGlobalScope, a separate JS realm from the rest of the
// app, so it's kept out of the main Vite/TypeScript bundle entirely.
//
// Buffers incoming render quanta, resamples to 16kHz if the AudioContext
// wasn't created at that rate, and posts ~200ms Float32 PCM frames back to
// the main thread — small increments so the backend's continuous streaming
// session can process audio close to real time.

const TARGET_SAMPLE_RATE = 16000;
const FRAME_SECONDS = 0.2;

function linearResample(input, ratio) {
  const outLength = Math.max(1, Math.floor(input.length / ratio));
  const output = new Float32Array(outLength);
  for (let i = 0; i < outLength; i++) {
    const srcIndex = i * ratio;
    const i0 = Math.floor(srcIndex);
    const i1 = Math.min(i0 + 1, input.length - 1);
    const frac = srcIndex - i0;
    output[i] = input[i0] * (1 - frac) + input[i1] * frac;
  }
  return output;
}

class PCMWorkletProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.chunks = [];
    this.bufferedSamples = 0;
    this.resampleRatio = sampleRate / TARGET_SAMPLE_RATE;
    this.frameSize = Math.round(TARGET_SAMPLE_RATE * FRAME_SECONDS);
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel && channel.length > 0) {
      const pcm = this.resampleRatio === 1 ? channel.slice() : linearResample(channel, this.resampleRatio);
      this.chunks.push(pcm);
      this.bufferedSamples += pcm.length;

      if (this.bufferedSamples >= this.frameSize) {
        const merged = new Float32Array(this.bufferedSamples);
        let offset = 0;
        for (const chunk of this.chunks) {
          merged.set(chunk, offset);
          offset += chunk.length;
        }
        this.port.postMessage(merged, [merged.buffer]);
        this.chunks = [];
        this.bufferedSamples = 0;
      }
    }
    return true;
  }
}

registerProcessor('pcm-worklet-processor', PCMWorkletProcessor);
