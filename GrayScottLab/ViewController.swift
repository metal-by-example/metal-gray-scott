import Cocoa
import Metal
import MetalKit

class ViewController: NSViewController, MTKViewDelegate {
    struct GrayScottPreset {
        let title: String
        let F: Float
        let k: Float
    }

    // Parameters based on R. Munafo's taxonomy of Pearson's patterns
    // http://www.mrob.com/pub/comp/xmorphia/pearson-classes.html
    let presets = [
        GrayScottPreset(title: "Custom", F: 0, k: 0),
        GrayScottPreset(title: "α", F: 0.014, k: 0.049),
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

    let context = GPUContext.default
    let simulationSize = MTLSize(width: 256, height: 256, depth: 1)

    @IBOutlet weak var mtkView: MTKView!
    @IBOutlet weak var fSlider: NSSlider!
    @IBOutlet weak var kSlider: NSSlider!
    @IBOutlet weak var fLabel: NSTextField!
    @IBOutlet weak var kLabel: NSTextField!
    @IBOutlet weak var presetPopUpButton: NSPopUpButton!
    
    private let renderer: Renderer
    private let simulation: GrayScottSimulation
    private let simulationStepsPerFrame = 20
    private let queueSemaphore = DispatchSemaphore(value: 1)
    private let simulationQueue = DispatchQueue(label: "com.metalbyexample.gray-scott", qos: .userInitiated)
    private let numberFormatter = NumberFormatter()

    required init?(coder: NSCoder) {
        numberFormatter.maximumFractionDigits = 4
        simulation = GrayScottSimulation(context: context, gridDimensions: simulationSize)
        renderer = Renderer(context: context, simulationSize: simulationSize)
        super.init(coder: coder)
    }

    override func viewDidLoad() {
        super.viewDidLoad()

        mtkView.device = context.device
        mtkView.delegate = self
        mtkView.colorPixelFormat = .bgra8Unorm_srgb
        mtkView.colorspace = CGColorSpace(name: CGColorSpace.sRGB)

        fSlider.floatValue = simulation.params.F
        kSlider.floatValue = simulation.params.K
        fLabel.stringValue = numberFormatter.string(from: NSNumber(value: simulation.params.F)) ?? "--"
        kLabel.stringValue = numberFormatter.string(from: NSNumber(value: simulation.params.K)) ?? "--"
        
        presetPopUpButton.removeAllItems()
        presetPopUpButton.addItems(withTitles: presets.map { $0.title })
        presetPopUpButton.selectItem(at: 8)
        presetSelectionChanged(self)
    }
    
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
    }

    func draw(in view: MTKView) {
        guard let renderPassDescriptor = mtkView.currentRenderPassDescriptor else { return }
        renderPassDescriptor.colorAttachments[0].loadAction = .dontCare

        if queueSemaphore.wait(timeout: .now()) == .success {
            simulationQueue.async {
                defer {
                    self.queueSemaphore.signal()
                }
                let dT: Float = 1.0
                if let resultTexture = self.simulation.perform(stepCount: self.simulationStepsPerFrame, timestep: dT) {
                    self.renderer.copySimulationResults(from: resultTexture)
                }
            }
        } else {
            print("Dropped simulation kick at time \(CACurrentMediaTime()) due to falling behind renderer")
        }

        guard let commandBuffer = self.context.commandQueue.makeCommandBuffer() else { return }
        defer {
            commandBuffer.commit()
        }

        if let renderCommandEncoder = commandBuffer.makeRenderCommandEncoder(descriptor: renderPassDescriptor) {
            self.renderer.draw(renderCommandEncoder)
            renderCommandEncoder.endEncoding()
        }

        if let drawable = mtkView.currentDrawable {
            commandBuffer.present(drawable, afterMinimumDuration: 1.0 / 60.0)
        }
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
        fLabel.stringValue = numberFormatter.string(from: NSNumber(value: params.F)) ?? "--"
        kLabel.stringValue = numberFormatter.string(from: NSNumber(value: params.K)) ?? "--"
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
