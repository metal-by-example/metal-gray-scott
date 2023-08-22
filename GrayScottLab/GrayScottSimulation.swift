
import Foundation
import Metal

class GrayScottSimulation {
    struct Params {
        var F: Float = 0.022
        var K: Float = 0.051
        let Du: Float = 2e-1
        let Dv: Float = 1e-1
    }

    let context: GPUContext
    let gridDimensions: MTLSize
    var params = Params()

    private let simulationPipelineIndex: Int
    private let seedPipelineIndex: Int

    let textureCount = 2
    private let textureSemaphore: DispatchSemaphore
    private var textures = [MTLTexture]()
    private var sourceTextureIndex = 0

    init(context: GPUContext, gridDimensions: MTLSize) {
        self.context = context
        self.gridDimensions = gridDimensions

        simulationPipelineIndex = context.cacheComputePipeline(for: "gray_scott")!
        seedPipelineIndex = context.cacheComputePipeline(for: "seed")!

        textureSemaphore = DispatchSemaphore(value: textureCount)

        for _ in 0..<textureCount {
            let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg32Float,
                                                                             width: gridDimensions.width,
                                                                             height: gridDimensions.height,
                                                                             mipmapped: false)
            textureDescriptor.storageMode = .private
            textureDescriptor.usage = [ .shaderRead, .shaderWrite ]
            let texture = context.device.makeTexture(descriptor: textureDescriptor)!
            textures.append(texture)
        }

        reseed()
    }

    func reseed(_ seedValue: Float = 0.0) {
        guard let commandBuffer = context.commandQueue.makeCommandBuffer() else { return }
        let destTexture = textures[sourceTextureIndex]

        let encoder = commandBuffer.makeComputeCommandEncoder()!

        let computePipeline = context.computePipeline(at: seedPipelineIndex)!
        encoder.setComputePipelineState(computePipeline)

        encoder.setTexture(destTexture, index: 0)

        var seed = seedValue
        encoder.setBytes(&seed, length: MemoryLayout.size(ofValue: seed), index: 0)

        let gridSize = gridDimensions
        let threadgroupSize = MTLSize(width: 8, height: 8, depth: 1)
        let threadgroupsPerGrid = MTLSize(width: (gridSize.width + threadgroupSize.width - 1) / threadgroupSize.width,
                                          height: (gridSize.height + threadgroupSize.height - 1) / threadgroupSize.height,
                                          depth: 1)

        encoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadgroupSize)
        encoder.endEncoding()

        commandBuffer.commit()
    }

    func perform(stepCount: Int, timestep dt: Float) -> MTLTexture? {
        var resultTexture: MTLTexture?
        guard let commandBuffer = context.commandQueue.makeCommandBuffer() else { return resultTexture }
        let commandEncoder = commandBuffer.makeComputeCommandEncoder()!
        for _ in 0..<stepCount {
            let sourceTexture = textures[sourceTextureIndex]
            let destTexture = textures[(sourceTextureIndex + 1) % textures.count]
            encodeSingleStep(commandEncoder, sourceTexture: sourceTexture, destTexture: destTexture, timestep: dt)
            sourceTextureIndex = (sourceTextureIndex + 1) % textures.count
            resultTexture = destTexture
        }
        commandEncoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        return resultTexture
    }

    private func encodeSingleStep(_ commandEncoder: MTLComputeCommandEncoder,
                                  sourceTexture: MTLTexture,
                                  destTexture: MTLTexture,
                                  timestep: Float)
    {
        let computePipeline = context.computePipeline(at: simulationPipelineIndex)!
        commandEncoder.setComputePipelineState(computePipeline)

        var stepParams = params
        commandEncoder.setBytes(&stepParams, length: MemoryLayout.size(ofValue: stepParams), index: 0)
        var dT = timestep
        commandEncoder.setBytes(&dT, length: MemoryLayout.size(ofValue: dT), index: 1)

        commandEncoder.setTextures([sourceTexture, destTexture], range: 0..<2)

        let gridSize = gridDimensions
        let threadgroupSize = MTLSize(width: 8, height: 8, depth: 1)
        let threadgroupsPerGrid = MTLSize(width: (gridSize.width + threadgroupSize.width - 1) / threadgroupSize.width,
                                          height: (gridSize.height + threadgroupSize.height - 1) / threadgroupSize.height,
                                          depth: 1)

        commandEncoder.dispatchThreadgroups(threadgroupsPerGrid, threadsPerThreadgroup: threadgroupSize)
    }
}
