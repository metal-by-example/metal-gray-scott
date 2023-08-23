import Metal

enum GPUContextError: Error {
    case noSuchFunction(_ functionName: String)
}

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
    
    func cacheComputePipeline(for functionName: String) throws -> Int {
        guard let function = defaultLibrary.makeFunction(name: functionName) else {
            throw GPUContextError.noSuchFunction(functionName)
        }

        let pipeline = try device.makeComputePipelineState(function: function)
        let pipelineIndex = computePipelineCache.count
        computePipelineCache.append(pipeline)
        return pipelineIndex
    }
    
    func computePipeline(at index: Int) -> MTLComputePipelineState {
        assert(index >= 0 && index < computePipelineCache.count)
        return computePipelineCache[index]
    }
    
    func cacheRenderPipeline(vertexFunction vertexFunctionName: String,
                             fragmentFunction fragmentFunctionName: String,
                             vertexDescriptor: MTLVertexDescriptor) throws -> Int
    {
        guard let vertexFunction = defaultLibrary.makeFunction(name: vertexFunctionName) else {
            throw GPUContextError.noSuchFunction(vertexFunctionName)
        }
        guard let fragmentFunction = defaultLibrary.makeFunction(name: fragmentFunctionName) else {
            throw GPUContextError.noSuchFunction(fragmentFunctionName)
        }

        let renderPipelineDescriptor = MTLRenderPipelineDescriptor()
        renderPipelineDescriptor.colorAttachments[0].pixelFormat = .bgra8Unorm_srgb
        renderPipelineDescriptor.vertexDescriptor = vertexDescriptor
        renderPipelineDescriptor.vertexFunction = vertexFunction
        renderPipelineDescriptor.fragmentFunction = fragmentFunction

        let renderPipelineState = try device.makeRenderPipelineState(descriptor: renderPipelineDescriptor)
        let pipelineIndex = renderPipelineCache.count
        renderPipelineCache.append(renderPipelineState)
        return pipelineIndex
    }

    func renderPipeline(at index: Int) -> MTLRenderPipelineState {
        assert(index >= 0 && index < renderPipelineCache.count)
        return renderPipelineCache[index]
    }
}
