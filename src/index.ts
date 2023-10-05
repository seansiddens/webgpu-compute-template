import computeShaderCode from './compute.wgsl';

async function main() {
    const canvas = document.querySelector('canvas')!;
    const context = canvas.getContext('webgpu');

    if (!context) {
        console.error("WebGPU is not supported!");
        throw new Error("WebGPU is not supported!");
        return;
    }

    console.log("Successfully created WebGPU context!");

    // Initial WebGPU setup
    const gpu = navigator.gpu;
    const adapter = await gpu.requestAdapter();
    if (!adapter) {
        throw new Error('WebGPU not supported.');
        return;
    }
    const device = await adapter.requestDevice();

    // Create buffers
    const bufferLength = 1024;
    const aData = new Float32Array([1, 2, 3, 4]);
    const bData = new Float32Array([1, 1, 1, 1]);
    const resultData = new Float32Array(bufferLength);

    const aBuffer = device.createBuffer({
        size: aData.byteLength,
        usage: GPUBufferUsage.STORAGE,
        mappedAtCreation: true,
    });
    new Float32Array(aBuffer.getMappedRange()).set(aData);
    aBuffer.unmap();

    const bBuffer = device.createBuffer({
        size: bData.byteLength,
        usage: GPUBufferUsage.STORAGE,
        mappedAtCreation: true,
    });
    new Float32Array(bBuffer.getMappedRange()).set(bData);
    bBuffer.unmap();

    const resultBuffer = device.createBuffer({
        size: resultData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    // Create compute pipeline.
    const computePipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
            module: device.createShaderModule({
                code: computeShaderCode,
            }),
            entryPoint: 'main',
        }
    });

    // Queue commands.
    const commandEncoder = device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(computePipeline);
    computePass.setBindGroup(0, device.createBindGroup({
        layout: computePipeline.getBindGroupLayout(0),
        entries: [
            {binding: 0, resource: {buffer: aBuffer}},
            {binding: 1, resource: {buffer: bBuffer}},
            {binding: 2, resource: {buffer: resultBuffer}},
        ],
    }));
    computePass.dispatchWorkgroups(bufferLength);
    computePass.end();

    // Get a GPU buffer for reading in an unmapped state.
    const gpuReadBuffer = device.createBuffer({
        size: resultData.byteLength,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    // Encode commands for copying to buffer.
    commandEncoder.copyBufferToBuffer(
        resultBuffer, // src
        0,
        gpuReadBuffer, // dst
        0,
        resultData.byteLength, // size
    );

    // Submit the commands.
    device.queue.submit([commandEncoder.finish()]);

    // Read the result back from the resultBuffer
    device.queue.onSubmittedWorkDone().then(async () => {
        await gpuReadBuffer.mapAsync(GPUMapMode.READ);
        const resultArray = new Float32Array(gpuReadBuffer.getMappedRange());
        console.log(resultArray);
        gpuReadBuffer.unmap();
    });

}

main();
