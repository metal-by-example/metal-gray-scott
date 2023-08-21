
import Foundation
import Metal
import MetalKit

class GPUContext {
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let defaultLibrary: MTLLibrary
    private var renderPipelineCache: [MTLRenderPipelineState] = []
    private var computePipelineCache: [MTLComputePipelineState] = []
    
    static let `default`: GPUContext = {
        guard let device = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal is not supported here")
        }
        let commandQueue = device.makeCommandQueue()!
        return GPUContext(device: device, commandQueue: commandQueue)
    }()
    
    init(device: MTLDevice, commandQueue: MTLCommandQueue) {
        self.device = device
        self.commandQueue = commandQueue
        guard let library = device.makeDefaultLibrary() else {
            fatalError("Could not create default Metal library from main bundle")
        }
        self.defaultLibrary = library
    }
    
    func cacheComputePipeline(for functionName: String) -> Int? {
        guard let function = defaultLibrary.makeFunction(name: functionName) else {
            fatalError("Could not find kernel function \"\(functionName)\" in default library")
        }
        do {
            let pipeline = try device.makeComputePipelineState(function: function)
            let pipelineIndex = computePipelineCache.count
            computePipelineCache.append(pipeline)
            return pipelineIndex
        } catch {
            fatalError("Could not create compute pipeline due to compiler error: \(error)")
        }
    }
    
    func computePipeline(at index: Int) -> MTLComputePipelineState? {
        return computePipelineCache[index]
    }
    
    func cacheRenderPipeline(vertexFunction vertexFunctionName: String,
                             fragmentFunction fragmentFunctionName: String,
                             vertexDescriptor: MTLVertexDescriptor) -> Int?
    {
        let vertexFunction = defaultLibrary.makeFunction(name: vertexFunctionName)!
        let fragmentFunction = defaultLibrary.makeFunction(name: fragmentFunctionName)!
        
        let renderPipelineDescriptor = MTLRenderPipelineDescriptor()
        renderPipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
        renderPipelineDescriptor.vertexDescriptor = vertexDescriptor
        renderPipelineDescriptor.vertexFunction = vertexFunction
        renderPipelineDescriptor.fragmentFunction = fragmentFunction
        
        do {
            let renderPipelineState = try device.makeRenderPipelineState(descriptor: renderPipelineDescriptor)
            let pipelineIndex = renderPipelineCache.count
            renderPipelineCache.append(renderPipelineState)
            return pipelineIndex
        } catch {
            fatalError("Could not create render pipeline state: \(error)")
        }
    }

    func renderPipeline(at index: Int) -> MTLRenderPipelineState? {
        return renderPipelineCache[index]
    }
}
