const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");

const source = fs.readFileSync(path.join(__dirname, "../web/r2t2-worklet.js"), "utf8");

function resample(sourceRate, frequency, seconds, options = {}) {
    const packets = [];
    let outputSamples = 0;
    let Processor;
    class AudioWorkletProcessor {
        constructor() {
            this.port = {onmessage: null, postMessage(message) {
                if (message.type === "pcm") {
                    outputSamples += message.samples.length;
                    if (options.collect !== false) packets.push(...message.samples);
                }
            }};
        }
    }
    vm.runInNewContext(source, {
        AudioWorkletProcessor, sampleRate: sourceRate,
        registerProcessor(name, type) { Processor = type; },
        Float32Array, Math,
    });
    const worklet = new Processor();
    const count = Math.round(sourceRate * seconds);
    for (let offset = 0, frame = 0; offset < count; frame++) {
        const size = options.randomFrames ? 64 + ((frame * 131 + 17) % 449) : 128;
        const length = Math.min(size, count - offset);
        const input = new Float32Array(length);
        if (frequency) {
            for (let i = 0; i < length; i++) {
                input[i] = Math.sin(2 * Math.PI * frequency * (offset + i) / sourceRate);
            }
        }
        worklet.process([[input]]);
        offset += length;
    }
    worklet.port.onmessage({data: {type: "flush"}});
    return options.collect === false ? outputSamples : packets;
}

function rms(values) {
    return Math.sqrt(values.reduce((sum, value) => sum + value * value, 0) / values.length);
}

for (const rate of [44100, 48000]) {
    const voice = resample(rate, 1000, 1.5);
    const alias = resample(rate, 10000, 1.5);
    const randomFrames = resample(rate, 1000, 1.5, {randomFrames: true});
    assert.equal(voice.length, 24000);
    assert.equal(alias.length, 24000);
    assert.equal(randomFrames.length, voice.length);
    assert.ok(voice.every((value, index) => Math.abs(value - randomFrames[index]) < 1e-6),
        `Frame boundaries changed output at ${rate}`);
    assert.ok(rms(voice) > 0.6, `1 kHz voice band attenuated at ${rate}`);
    assert.ok(rms(alias) < 0.08, `10 kHz alias insufficiently filtered at ${rate}`);
}
assert.equal(resample(44100, 0, 600, {randomFrames: true, collect: false}), 9600000);
console.log("AudioWorklet resampler: duration, frame-boundary, 10-minute drift, and anti-alias checks passed");
