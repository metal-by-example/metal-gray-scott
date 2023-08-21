
import Foundation
import Metal

class Renderer {
    let context: GPUContext
    var viewportSize = CGSize.zero

    var simulationTexture: MTLTexture?

    var vertexDescriptor = MTLVertexDescriptor()
    var renderPipelineIndex: Int = -1
    var vertexBuffer: MTLBuffer!

    init(context: GPUContext) {
        self.context = context
        makeResources()
        makePipelines()
    }

    func makeResources() {
        let vertexData: [Float] = [
        //    x     y    z    u    v
            -1.0,  1.0, 0.0, 0.0, 0.0, // upper left
            -1.0, -1.0, 0.0, 0.0, 1.0, // lower left
             1.0,  1.0, 0.0, 1.0, 0.0, // upper right
             1.0, -1.0, 0.0, 1.0, 1.0, // lower right
        ]

        vertexDescriptor.attributes[0].format = .float3
        vertexDescriptor.attributes[0].offset = 0
        vertexDescriptor.attributes[0].bufferIndex = 0

        vertexDescriptor.attributes[1].format = .float2
        vertexDescriptor.attributes[1].offset = MemoryLayout<Float>.stride * 3
        vertexDescriptor.attributes[1].bufferIndex = 0

        vertexDescriptor.layouts[0].stride = MemoryLayout<Float>.stride * 5

        vertexBuffer = context.device.makeBuffer(bytes: vertexData,
                                                 length: MemoryLayout<Float>.stride * vertexData.count,
                                                 options: [.storageModeShared])!
    }

    func makePipelines() {
        renderPipelineIndex = context.cacheRenderPipeline(vertexFunction: "vertex_main",
                                                          fragmentFunction: "fragment_main",
                                                          vertexDescriptor: vertexDescriptor)!
    }

    func draw(_ renderCommandEncoder: MTLRenderCommandEncoder) {
        let renderPipelineState = context.renderPipeline(at: renderPipelineIndex)!
        guard let texture = simulationTexture else { return }
        renderCommandEncoder.setFrontFacing(.counterClockwise)
        renderCommandEncoder.setCullMode(.back)
        renderCommandEncoder.setRenderPipelineState(renderPipelineState)
        renderCommandEncoder.setVertexBuffer(vertexBuffer, offset: 0, index: 0)
        renderCommandEncoder.setFragmentTexture(texture, index: 0)
        renderCommandEncoder.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    }
}
