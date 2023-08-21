import Cocoa
import MetalKit

class ViewController: NSViewController, MTKViewDelegate {
    struct GrayScottPreset {
        let title: String
        let F: Float
        let k: Float
    }

    let context = GPUContext.default

    @IBOutlet weak var mtkView: MTKView!
    @IBOutlet weak var fSlider: NSSlider!
    @IBOutlet weak var kSlider: NSSlider!
    @IBOutlet weak var fLabel: NSTextField!
    @IBOutlet weak var kLabel: NSTextField!
    @IBOutlet weak var presetPopUpButton: NSPopUpButton!
    
    private var renderer: Renderer!
    private var simulation: GrayScottSimulation!

    // Parameters based on R. Munafo's taxonomy of Pearson's patterns
    // http://www.mrob.com/pub/comp/xmorphia/pearson-classes.html
    private let presets = [
        GrayScottPreset(title: "Custom", F: 0, k: 0),
        GrayScottPreset(title: "α", F: 0.014, k: 0.053),
        GrayScottPreset(title: "β", F: 0.026, k: 0.052),
        GrayScottPreset(title: "γ", F: 0.026, k: 0.055),
        GrayScottPreset(title: "δ", F: 0.042, k: 0.059),
        GrayScottPreset(title: "ε", F: 0.018, k: 0.055),
        GrayScottPreset(title: "ζ", F: 0.026, k: 0.059),
        GrayScottPreset(title: "η", F: 0.034, k: 0.063),
        GrayScottPreset(title: "θ", F: 0.030, k: 0.057),
        GrayScottPreset(title: "ι", F: 0.046, k: 0.0594),
        GrayScottPreset(title: "κ", F: 0.050, k: 0.063),
        GrayScottPreset(title: "λ", F: 0.026, k: 0.061),
        GrayScottPreset(title: "μ", F: 0.046, k: 0.065),
    ]

    private let maxCommandBuffersInFlight = 63 // Exceeding this can cause total GPU lock-up and window server hang
    private let queueSemaphore = DispatchSemaphore(value: 1)
    private let simulationQueue = DispatchQueue(label: "com.metalbyexample.gray-scott", qos: .userInitiated)

    private let numberFormatter = NumberFormatter()

    override func viewDidLoad() {
        super.viewDidLoad()

        numberFormatter.maximumFractionDigits = 3

        let simulationSize = MTLSize(width: 256, height: 256, depth: 1)
        simulation = GrayScottSimulation(context: context, gridDimensions: simulationSize)

        renderer = Renderer(context: context)

        mtkView.device = context.device
        mtkView.delegate = self
        mtkView.colorPixelFormat = .bgra8Unorm_srgb
        mtkView.colorspace = CGColorSpace(name: CGColorSpace.sRGB)!

        fSlider.floatValue = simulation.params.F
        kSlider.floatValue = simulation.params.K
        fLabel.stringValue = numberFormatter.string(from: NSNumber(value: simulation.params.F)) ?? "--"
        kLabel.stringValue = numberFormatter.string(from: NSNumber(value: simulation.params.K)) ?? "--"
        presetPopUpButton.removeAllItems()
        presetPopUpButton.addItems(withTitles: presets.map { $0.title })

        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg32Float,
                                                                         width: simulationSize.width,
                                                                         height: simulationSize.height,
                                                                         mipmapped: false)
        textureDescriptor.storageMode = .private
        textureDescriptor.usage = .shaderRead
        renderer.simulationTexture = context.device.makeTexture(descriptor: textureDescriptor)

        presetPopUpButton.selectItem(at: 8)
        presetSelectionChanged(self)
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
    }

    func draw(in view: MTKView) {
        let stepsPerFrame = min(maxCommandBuffersInFlight, 30)
        let dT: Float = 1.0
        
        if queueSemaphore.wait(timeout: .now()) == .success {
            simulationQueue.async {
                defer {
                    self.queueSemaphore.signal()
                }
                let resultTexture = self.simulation.perform(stepCount: stepsPerFrame, timestep: dT)
                if let resultTexture {
                    guard let commandBuffer = self.context.commandQueue.makeCommandBuffer() else { return }
                    let blitEncoder = commandBuffer.makeBlitCommandEncoder()!
                    blitEncoder.copy(from: resultTexture, to: self.renderer.simulationTexture!)
                    blitEncoder.endEncoding()
                    commandBuffer.commit()
                }
            }
        } else {
            //print("Dropped simulation kick at time \(CACurrentMediaTime()) due to falling behind renderer")
        }

        guard let commandBuffer = self.context.commandQueue.makeCommandBuffer() else { return }
        guard let renderPassDescriptor = mtkView.currentRenderPassDescriptor else { return }
        let renderCommandEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor)!
        self.renderer.draw(renderCommandEncoder)
        renderCommandEncoder.endEncoding()
        commandBuffer.present(mtkView.currentDrawable!, afterMinimumDuration: 1.0 / 60.0)
        commandBuffer.commit()
    }

    func setSlidersFromPreset(at index: Int) {
        let preset = presets[index]
        if index != 0 { // Don't update sliders if "Custom" was selected
            fSlider.floatValue = preset.F
            kSlider.floatValue = preset.k
        }
        setSimulationParamsFromSliders()
    }

    func setSimulationParamsFromSliders() {
        var params = simulation.params
        params.F = fSlider.floatValue
        params.K = kSlider.floatValue
        fLabel.stringValue = numberFormatter.string(from: NSNumber(value: simulation.params.F)) ?? "--"
        kLabel.stringValue = numberFormatter.string(from: NSNumber(value: simulation.params.K)) ?? "--"
        simulation.params = params
    }

    @IBAction func sliderValueChanged(_ sender: Any) {
        presetPopUpButton.selectItem(at: 0)
        setSimulationParamsFromSliders()
    }

    @IBAction func presetSelectionChanged(_ sender: Any) {
        setSlidersFromPreset(at: presetPopUpButton.indexOfSelectedItem)
    }
    
    @IBAction func reseedButtonWasPressed(_ sender: Any) {
        let seed = Float(fmod(CACurrentMediaTime(), 1.0))
        simulation.reseed(seed)
    }
}
