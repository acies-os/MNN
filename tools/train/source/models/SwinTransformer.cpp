//
//  SwinTransformer.cpp
//  MNN
//




/**
 * Error: 
 * 1. _Gelu is undefined
 * 2. 
 */
#include <algorithm>
#include "SwinTransformer.hpp"
#include "SwinUtils.hpp"
#include "NeuralNetworkOp.hpp"
#include <MNN/expr/MathOp.hpp>
#include "MNN_generated.h"

namespace MNN {
namespace Train {
namespace Model {
using namespace MNN::Express;

// Helper function to create LayerNorm
static VARP _LayerNorm(VARP x, std::vector<int32_t> axis, float epsilon = 1e-5, std::vector<float> gamma = {}, std::vector<float> beta = {}) {
    std::unique_ptr<OpT> op(new OpT);
    op->main.type                         = OpParameter_LayerNorm;
    op->type                              = OpType_LayerNorm;
    op->main.value                        = new LayerNormT;
    if(gamma.size() != 0){
        op->main.AsLayerNorm()->gamma         = gamma;
    }
    if(beta.size() != 0){
        op->main.AsLayerNorm()->beta          = beta;
    }
    op->main.AsLayerNorm()->epsilon       = epsilon;
    op->main.AsLayerNorm()->axis          = axis;
    return (Variable::create(Expr::create(std::move(op), {x})));
}

// Helper function to create GELU activation
static VARP _Gelu(VARP x) {
    std::unique_ptr<OpT> op(new OpT);
    op->type = OpType_UnaryOp;
    op->main.type = OpParameter_UnaryOp;
    op->main.value = new UnaryOpT;
    op->main.AsUnaryOp()->opType = (UnaryOpOperation)32; // UnaryOpOperation_GELU
    return Variable::create(Expr::create(std::move(op), {x}));
}

// ============================================================================
// Mlp Implementation
// ============================================================================
Mlp::Mlp(int inFeatures, int hiddenFeatures, int outFeatures, float drop) {
    if (outFeatures == -1) outFeatures = inFeatures;
    if (hiddenFeatures == -1) hiddenFeatures = inFeatures;
    
    fc1.reset(NN::Linear(inFeatures, hiddenFeatures));
    fc2.reset(NN::Linear(hiddenFeatures, outFeatures));
    dropout.reset(NN::Dropout(drop));
    
    registerModel({fc1, fc2, dropout});
}

std::vector<VARP> Mlp::onForward(const std::vector<VARP> &inputs) {
    VARP x = inputs[0];
    
    x = fc1->forward(x);
    x = _Gelu(x);
    x = dropout->forward(x);
    x = fc2->forward(x);
    x = dropout->forward(x);
    
    return {x};
}

// ============================================================================
// WindowAttention Implementation
// ============================================================================
WindowAttention::WindowAttention(int dim, const std::vector<int>& windowSize, int numHeads,
                                bool qkvBias, float qkScale, float attnDrop, float projDrop) 
    : dim(dim), windowSize(windowSize), numHeads(numHeads) {
    
    int headDim = dim / numHeads;
    scale = (qkScale > 0) ? qkScale : 1.0f / _Sqrt(_Scalar<float>(static_cast<float>(headDim)))->readMap<float>()[0];
    
    // Create relative position bias table
    relativePositionBiasTable = createRelativePositionBiasTable(windowSize, numHeads);
    relativePositionIndex = getRelativePositionIndex(windowSize);
    
    qkv.reset(NN::Linear(dim, dim * 3, qkvBias));
    attnDropout.reset(NN::Dropout(attnDrop));
    proj.reset(NN::Linear(dim, dim));
    projDropout.reset(NN::Dropout(projDrop));
    
    registerModel({qkv, attnDropout, proj, projDropout});
}

std::vector<VARP> WindowAttention::onForward(const std::vector<VARP> &inputs) {
    VARP x = inputs[0];
    VARP mask = inputs.size() > 1 ? inputs[1] : nullptr;
    
    auto info = x->getInfo();
    int B_ = info->dim[0];  // num_windows * B
    int N = info->dim[1];   // window_size * window_size
    int C = info->dim[2];   // channels
    
    // QKV projection
    VARP qkvOut = qkv->forward(x);
    qkvOut = _Reshape(qkvOut, {B_, N, 3, numHeads, C / numHeads});
    qkvOut = _Transpose(qkvOut, {2, 0, 3, 1, 4});
    
    // Split Q, K, V
    auto qkvSplit = _Split(qkvOut, {1, 1, 1}, 0);
    VARP q = _Squeeze(qkvSplit[0], {0}) * _Scalar<float>(scale);
    VARP k = _Squeeze(qkvSplit[1], {0});
    VARP v = _Squeeze(qkvSplit[2], {0});
    
    // Attention computation
    VARP attn = _MatMul(q, _Transpose(k, {0, 1, 3, 2}));
    
    // Add relative position bias - simplified implementation
    // In actual implementation, proper indexing operations would be used
    // For now, just use the attention without relative position bias
    // attn = attn + relativePosBias;
    
    // Apply mask if provided
    if (mask == nullptr) {
        
    } else {
        attn = attn + mask;
    }
    
    attn = _Softmax(attn, -1);
    attn = attnDropout->forward(attn);
    
    // Apply attention to values
    x = _MatMul(attn, v);
    x = _Transpose(x, {0, 2, 1, 3});
    x = _Reshape(x, {B_, N, C});
    
    // Final projection
    x = proj->forward(x);
    x = projDropout->forward(x);
    
    return {x};
}

// ============================================================================
// SwinTransformerBlock Implementation
// ============================================================================
SwinTransformerBlock::SwinTransformerBlock(int dim, const std::vector<int>& inputResolution, 
                                            int numHeads, const std::vector<int>& windowSize,
                                            const std::vector<int>& shiftSize, float mlpRatio,
                                            bool qkvBias, float qkScale, float drop, 
                                            float attnDrop, float dropPath)
    : dim(dim), inputResolution(inputResolution), numHeads(numHeads), 
        windowSize(windowSize), shiftSize(shiftSize), mlpRatio(mlpRatio), dropPathRate(dropPath) {
    
    // Adjust window size if larger than input resolution
    std::vector<int> adjustedWindowSize = windowSize;
    std::vector<int> adjustedShiftSize = shiftSize;
    
    if (inputResolution[0] <= windowSize[0]) {
        adjustedShiftSize[0] = 0;
        adjustedWindowSize[0] = inputResolution[0];
    }
    if (inputResolution[1] <= windowSize[1]) {
        adjustedShiftSize[1] = 0;
        adjustedWindowSize[1] = inputResolution[1];
    }
    
    this->windowSize = adjustedWindowSize;
    this->shiftSize = adjustedShiftSize;
    
    // norm1.reset(NN::LayerNorm({dim}));
    attn.reset(new WindowAttention(dim, this->windowSize, numHeads, qkvBias, qkScale, attnDrop, drop));
    // norm2.reset(NN::LayerNorm({dim}));
    
    int mlpHiddenDim = static_cast<int>(dim * mlpRatio);
    mlp.reset(new Mlp(dim, mlpHiddenDim, dim, drop));
    
    // Create attention mask for shifted window attention
    if (adjustedShiftSize[0] > 0 || adjustedShiftSize[1] > 0) {
        attnMask = createShiftedWindowMask(inputResolution, this->windowSize, this->shiftSize);
    }
    
    registerModel({norm1, norm2});
}

std::vector<VARP> SwinTransformerBlock::onForward(const std::vector<VARP> &inputs) {
    VARP x = inputs[0];
    int H = inputResolution[0];
    int W = inputResolution[1];
    
    auto info = x->getInfo();
    int B = info->dim[0];
    int L = info->dim[1];
    int C = info->dim[2];
    
    VARP shortcut = x;
    x = norm1->forward(x);
    x = _Reshape(x, {B, H, W, C});
    
    // Cyclic shift
    if (shiftSize[0] > 0 || shiftSize[1] > 0) {
        x = rollTensor(x, {-shiftSize[0], -shiftSize[1]}, {1, 2});
    }
    
    // Window partition
    VARP xWindows = windowPartition(x, windowSize);
    xWindows = _Reshape(xWindows, {-1, windowSize[0] * windowSize[1], C});
    
    // Window attention
    std::vector<VARP> attnInputs = {xWindows};
    if (attnMask == nullptr) {
    } else {
            attnInputs.push_back(attnMask);
    }
    VARP attnWindows = attn->onForward(attnInputs)[0];
    
    // Merge windows
    attnWindows = _Reshape(attnWindows, {-1, windowSize[0], windowSize[1], C});
    VARP shiftedX = windowReverse(attnWindows, windowSize, H, W);
    
    // Reverse cyclic shift
    if (shiftSize[0] > 0 || shiftSize[1] > 0) {
        x = rollTensor(shiftedX, {shiftSize[0], shiftSize[1]}, {1, 2});
    } else {
        x = shiftedX;
    }
    
    x = _Reshape(x, {B, H * W, C});
    
    // Apply drop path and residual connection
    bool isTraining = false; // TODO: Get training state from module
    x = dropPath(x, dropPathRate, isTraining);
    x = shortcut + x;
    
    // MLP
    shortcut = x;
    x = norm2->forward(x);
    x = mlp->forward(x);
    x = dropPath(x, dropPathRate, isTraining);
    x = shortcut + x;
    
    return {x};
}

// ============================================================================
// PatchMerging Implementation
// ============================================================================
PatchMerging::PatchMerging(const std::vector<int>& inputResolution, int dim)
    : inputResolution(inputResolution), dim(dim) {
    
    reduction.reset(NN::Linear(4 * dim, 2 * dim));
    // norm.reset(NN::LayerNorm({4 * dim}));
    
    registerModel({reduction, norm});
}

std::vector<VARP> PatchMerging::onForward(const std::vector<VARP> &inputs) {
    VARP x = inputs[0];
    int H = inputResolution[0];
    int W = inputResolution[1];
    
    auto info = x->getInfo();
    int B = info->dim[0];
    int L = info->dim[1];
    int C = info->dim[2];
    
    x = _Reshape(x, {B, H, W, C});
    
    // Extract patches at 2x2 stride - simplified implementation
    // In actual MNN implementation, you would use proper slicing operations
    auto x_reshaped = _Reshape(x, {B, H, W, C});
    
    // Simple downsampling by taking every other pixel
    // This is a placeholder - actual implementation would need proper strided slicing
    x = _Reshape(x_reshaped, {B, -1, 4 * C});
    
    x = norm->forward(x);
    x = reduction->forward(x);
    
    return {x};
}

// ============================================================================
// BasicLayer Implementation
// ============================================================================
BasicLayer::BasicLayer(int dim, const std::vector<int>& inputResolution, int depth, int numHeads,
                        const std::vector<int>& windowSize, float mlpRatio, bool qkvBias, 
                        float qkScale, float drop, float attnDrop, const std::vector<float>& dropPath,
                        bool useDownsample)
    : dim(dim), inputResolution(inputResolution), depth(depth) {
    
    // Build blocks
    for (int i = 0; i < depth; i++) {
        std::vector<int> shiftSize = {0, 0};
        if (i % 2 == 1) {  // Shifted window for odd blocks
            shiftSize = {windowSize[0] / 2, windowSize[1] / 2};
        }
        
        float blockDropPath = dropPath.empty() ? 0.0f : dropPath[i];
        
        auto block = std::make_shared<SwinTransformerBlock>(
            dim, inputResolution, numHeads, windowSize, shiftSize,
            mlpRatio, qkvBias, qkScale, drop, attnDrop, blockDropPath);
        blocks.push_back(block);
    }
    
    // Patch merging layer
    if (useDownsample) {
        downsample.reset(new PatchMerging(inputResolution, dim));
    }
    
    registerModel(std::vector<std::shared_ptr<Express::Module>>(blocks.begin(), blocks.end()));
    if (downsample) {
        registerModel({downsample});
    }
}

std::vector<VARP> BasicLayer::onForward(const std::vector<VARP> &inputs) {
    VARP x = inputs[0];
    
    for (auto& block : blocks) {
        x = block->forward(x);
    }
    
    if (downsample) {
        x = downsample->forward(x);
    }
    
    return {x};
}

// ============================================================================
// PatchEmbed Implementation
// ============================================================================
PatchEmbed::PatchEmbed(const std::vector<int>& imgSize, const std::vector<int>& patchSize,
                        int inChannels, int embedDim)
    : imgSize(imgSize), patchSize(patchSize), inChannels(inChannels), embedDim(embedDim) {
    
    patchesResolution = {imgSize[0] / patchSize[0], imgSize[1] / patchSize[1]};
    numPatches = patchesResolution[0] * patchesResolution[1];
    
    NN::ConvOption convOption;
    convOption.kernelSize = {patchSize[0], patchSize[1]};
    convOption.channel = {inChannels, embedDim};
    convOption.stride = {patchSize[0], patchSize[1]};
    convOption.padMode = Express::VALID;
    convOption.depthwise = false;
    
    proj.reset(NN::Conv(convOption, false));
    // norm.reset(NN::LayerNorm({embedDim}));
    
    registerModel({proj, norm});
}

std::vector<VARP> PatchEmbed::onForward(const std::vector<VARP> &inputs) {
    VARP x = inputs[0];
    
    auto info = x->getInfo();
    int B = info->dim[0];
    int C = info->dim[1];
    int H = info->dim[2];
    int W = info->dim[3];
    
    // Check input size
    assert(H == imgSize[0] && W == imgSize[1]);
    
    x = proj->forward(x);  // B embedDim Ph Pw
    x = _Reshape(x, {B, embedDim, -1});  // B embedDim Ph*Pw
    x = _Transpose(x, {0, 2, 1});  // B Ph*Pw embedDim
    x = norm->forward(x);
    
    return {x};
}

// ============================================================================
// SwinTransformerV4_CMC Implementation (Multi-modal Sensor Data)
// ============================================================================
SwinTransformerV4_CMC::SwinTransformerV4_CMC(int numClasses, int numSegments,
                                            const std::vector<std::string>& modalities,
                                            const std::vector<std::string>& locations,
                                            float dropRate, float dropPathRate, float attnDropRate,
                                            int timeFreqOutChannels, int timeFreqHeadNum,
                                            const std::map<std::string, std::vector<int>>& timeFreqBlockNum,
                                            int locOutChannels, float mlpRatio, bool qkvBias,
                                            bool apeEnabled, const std::map<std::string, std::vector<int>>& windowSize,
                                            const std::map<std::string, std::vector<int>>& patchSizeFreq,
                                            const std::map<std::string, std::map<std::string, int>>& locModInFreqChannels,
                                            const std::map<std::string, std::map<std::string, int>>& locModSpectrumLen,
                                            const std::map<std::string, int>& inStride,
                                            int fcDim, const std::string& pretrainedHead)
    : numClasses(numClasses), numSegments(numSegments), modalities(modalities), locations(locations),
        dropRate(dropRate), dropPathRate(dropPathRate), attnDropRate(attnDropRate),
        timeFreqOutChannels(timeFreqOutChannels), timeFreqHeadNum(timeFreqHeadNum),
        timeFreqBlockNum(timeFreqBlockNum), locOutChannels(locOutChannels), mlpRatio(mlpRatio),
        qkvBias(qkvBias), apeEnabled(apeEnabled), windowSize(windowSize), patchSizeFreq(patchSizeFreq),
        locModInFreqChannels(locModInFreqChannels), locModSpectrumLen(locModSpectrumLen),
        inStride(inStride), fcDim(fcDim), pretrainedHead(pretrainedHead) {
    
    // normLayer will be initialized as needed for specific dimensions
    normLayer = nullptr;
    
    // Initialize encoder components
    initEncoder();
    
    sampleDim = locOutChannels * modalities.size();
    
    // Classification layers
    if (pretrainedHead == "linear") {
        // Linear classification for supervised learning or finetuning
        classLayer.reset(NN::Linear(sampleDim, numClasses));
        detectionClassLayer.reset(NN::Linear(sampleDim, 2));
    } else {
        // For non-linear classification, we'll use separate layers
        // and handle the composition in the forward pass
        auto fc1 = NN::Linear(sampleDim, fcDim);
        auto fc2 = NN::Linear(fcDim, numClasses);
        // classLayer = fc2;  // Store the final layer, handle fc1 separately
        classLayer.reset(fc2);
        
        auto detectionFc1 = NN::Linear(sampleDim, fcDim);
        auto detectionFc2 = NN::Linear(fcDim, 2);
        // detectionClassLayer = detectionFc2;  // Store the final layer
        detectionClassLayer.reset(detectionFc2);
    }
    
    // Register all modules
    std::vector<std::shared_ptr<Express::Module>> modules = {classLayer, detectionClassLayer, normLayer};
    
    // Register patch embedding modules
    for (const auto& loc : locations) {
        for (const auto& mod : modalities) {
            modules.push_back(patchEmbed[loc][mod]);
            modules.push_back(modInLayers[loc][mod]);
            
            // Register frequency interval layers
            for (auto& layer : freqIntervalLayers[loc][mod]) {
                modules.push_back(layer);
            }
        }
    }
    
    registerModel(modules);
}

void SwinTransformerV4_CMC::initEncoder() {
    for (const auto& loc : locations) {
        freqIntervalLayers[loc] = {};
        patchEmbed[loc] = {};
        absolutePosEmbed[loc] = {};
        modInLayers[loc] = {};
        imgSizes[loc] = {};
        
        for (const auto& mod : modalities) {
            // Calculate image size for this modality
            int stride = inStride.at(mod);
            int spectrumLen = locModSpectrumLen.at(loc).at(mod);
            std::vector<int> imgSize = {numSegments, spectrumLen / stride};
            
            // Get padded image size
            std::vector<int> paddedImgSize = getPaddedSize(imgSize, windowSize.at(mod), 
                                                            patchSizeFreq.at(mod), 
                                                            timeFreqBlockNum.at(mod).size());
            imgSizes[loc][mod] = paddedImgSize;
            
            // Patch embedding
            int inChannels = locModInFreqChannels.at(loc).at(mod) * stride;
            patchEmbed[loc][mod] = std::make_shared<PatchEmbed>(paddedImgSize, patchSizeFreq.at(mod),
                                                                inChannels, timeFreqOutChannels);
            
            auto patchesResolution = patchEmbed[loc][mod]->getPatchesResolution();
            
            // Absolute positional embedding
            if (apeEnabled) {
                absolutePosEmbed[loc][mod] = _TrainableParam(0.0f, {1, patchEmbed[loc][mod]->getNumPatches(), 
                                                                    timeFreqOutChannels}, NCHW);
                truncNormal(absolutePosEmbed[loc][mod], 0.0f, 0.02f, -2.0f, 2.0f);
            }
            
            // Swin Transformer blocks
            freqIntervalLayers[loc][mod] = {};
            
            // Drop path rate schedule
            int totalBlocks = std::accumulate(timeFreqBlockNum.at(mod).begin(), 
                                            timeFreqBlockNum.at(mod).end(), 0);
            std::vector<float> dpr;
            for (int i = 0; i < totalBlocks; i++) {
                dpr.push_back(dropPathRate * i / (totalBlocks - 1));
            }
            
            int dprIndex = 0;
            std::vector<int> currentResolution = patchesResolution;
            int currentDim = timeFreqOutChannels;
            
            for (size_t iLayer = 0; iLayer < timeFreqBlockNum.at(mod).size(); iLayer++) {
                int downRatio = 1 << iLayer;
                int layerDim = timeFreqOutChannels * downRatio;
                
                std::vector<int> layerResolution = {patchesResolution[0] / downRatio,
                                                    patchesResolution[1] / downRatio};
                
                std::vector<float> layerDropPath(dpr.begin() + dprIndex,
                                                dpr.begin() + dprIndex + timeFreqBlockNum.at(mod)[iLayer]);
                dprIndex += timeFreqBlockNum.at(mod)[iLayer];
                
                bool useDownsample = (iLayer < timeFreqBlockNum.at(mod).size() - 1);
                
                auto layer = std::make_shared<BasicLayer>(layerDim, layerResolution,
                                                            timeFreqHeadNum, windowSize.at(mod),
                                                            mlpRatio, qkvBias, -1.0f, dropRate,
                                                            attnDropRate, layerDropPath, useDownsample);
                
                freqIntervalLayers[loc][mod].push_back(layer);
                currentDim = layerDim * (useDownsample ? 2 : 1);
            }
            
            // Modality input layer to unify channels
            int finalFeatureSize = (patchesResolution[0] / (1 << (timeFreqBlockNum.at(mod).size() - 1))) *
                                  (patchesResolution[1] / (1 << (timeFreqBlockNum.at(mod).size() - 1))) *
                                    currentDim;
            
            modInLayers[loc][mod].reset(NN::Linear(finalFeatureSize, locOutChannels));
        }
    }
}

std::pair<VARP, std::vector<int>> SwinTransformerV4_CMC::padInput(VARP freqInput, 
                                                                    const std::string& loc, 
                                                                    const std::string& mod) {
    int stride = inStride.at(mod);
    int spectrumLen = locModSpectrumLen.at(loc).at(mod);
    std::vector<int> imgSize = {numSegments, spectrumLen / stride};
    
    auto info = freqInput->getInfo();
    int b = info->dim[0];
    int c = info->dim[1];
    int i = info->dim[2];
    int s = info->dim[3];
    
    // [b, c, i, spectrum] -> [b, i, spectrum, c]
    freqInput = _Transpose(freqInput, {0, 2, 3, 1});
    
    // Reshape to reduce stride dimension
    freqInput = _Reshape(freqInput, {b, i, s / stride, c * stride});
    
    // [b, i, spectrum, c] -> [b, c, i, spectrum]
    freqInput = _Transpose(freqInput, {0, 3, 1, 2});
    
    // Pad to required size - simplified implementation
    std::vector<int> paddedImgSize = imgSizes[loc][mod];
    int padHeight = paddedImgSize[0] - imgSize[0];
    int padWidth = paddedImgSize[1] - imgSize[1];
    
    // For now, return the reshaped input without padding
    // In actual implementation, proper padding operation would be used
    if (padHeight > 0 || padWidth > 0) {
        // Use identity operation for now - replace with actual padding in production
        freqInput = freqInput;
    }
    
    return {freqInput, paddedImgSize};
}

std::map<std::string, std::map<std::string, VARP>> SwinTransformerV4_CMC::patchForward(
    const std::map<std::string, std::map<std::string, VARP>>& freqX) {
    
    std::map<std::string, std::map<std::string, VARP>> embeddedInputs;
    
    for (const auto& loc : locations) {
        embeddedInputs[loc] = {};
        for (const auto& mod : modalities) {
            // Pad input and apply patch embedding
            auto [paddedInput, paddedSize] = padInput(freqX.at(loc).at(mod), loc, mod);
            VARP embeddedInput = patchEmbed[loc][mod]->forward(paddedInput);
            embeddedInputs[loc][mod] = embeddedInput;
        }
    }
    
    return embeddedInputs;
}

std::vector<VARP> SwinTransformerV4_CMC::forwardEncoder(
    const std::map<std::string, std::map<std::string, VARP>>& patchedInputs) {
    
    // Feature extraction for each modality and location
    std::map<std::string, std::vector<VARP>> modLocFeatures;
    for (const auto& mod : modalities) {
        modLocFeatures[mod] = {};
    }
    
    for (const auto& loc : locations) {
        for (const auto& mod : modalities) {
            VARP embeddedInput = patchedInputs.at(loc).at(mod);
            auto info = embeddedInput->getInfo();
            int b = info->dim[0];
            
            // Add absolute positional embedding if enabled
            if (apeEnabled && !(absolutePosEmbed[loc][mod] == nullptr)) {
                embeddedInput = embeddedInput + absolutePosEmbed[loc][mod];
            }
            
            // Forward through Swin Transformer layers
            VARP freqIntervalOutput = embeddedInput;
            for (auto& layer : freqIntervalLayers[loc][mod]) {
                freqIntervalOutput = layer->forward(freqIntervalOutput);
            }
            
            // Unify input channels for this modality
            freqIntervalOutput = _Reshape(freqIntervalOutput, {b, -1});
            freqIntervalOutput = modInLayers[loc][mod]->forward(freqIntervalOutput);
            freqIntervalOutput = _Reshape(freqIntervalOutput, {b, 1, -1});
            
            modLocFeatures[mod].push_back(freqIntervalOutput);
        }
    }
    
    // Concatenate location features for each modality
    std::vector<VARP> modFeatures;
    for (const auto& mod : modalities) {
        if (modLocFeatures[mod].size() > 1) {
            VARP modFeature = _Stack(modLocFeatures[mod], 2);
            modFeature = _Reshape(modFeature, {modFeature->getInfo()->dim[0], -1});
            modFeatures.push_back(modFeature);
        } else {
            modFeatures.push_back(_Reshape(modLocFeatures[mod][0], {modLocFeatures[mod][0]->getInfo()->dim[0], -1}));
        }
    }
    
    // Concatenate all modality features
    VARP sampleFeatures = _Concat(modFeatures, 1);
    VARP logits = classLayer->forward(sampleFeatures);
    
    return {logits};
}

std::vector<VARP> SwinTransformerV4_CMC::onForward(const std::vector<VARP> &inputs) {
    // Expecting seismic and audio inputs
    VARP seismicX = inputs[0];
    VARP audioX = inputs[1];
    
    // Organize inputs by location and modality
    std::map<std::string, std::map<std::string, VARP>> freqX;
    freqX[locations[0]]["seismic"] = seismicX;
    freqX[locations[0]]["audio"] = audioX;
    
    // Patch embedding
    auto patchedInputs = patchForward(freqX);
    
    // Forward through encoder
    auto logits = forwardEncoder(patchedInputs);
    
    return logits;
}

} // namespace Model
} // namespace Train
} // namespace MNN