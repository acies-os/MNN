//
//  SwinTransformer.hpp
//  MNN
//
//  Created by MNN on 2025/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef SwinTransformer_hpp
#define SwinTransformer_hpp

#include "Initializer.hpp"
#include <vector>
#include <string>
#include <map>
#include <numeric>
#include "SwinUtils.hpp"
#include <MNN/expr/Module.hpp>
#include "NN.hpp"
#include <algorithm>
#include <MNN/expr/Optimizer.hpp>
#include "OpConverter.hpp"
#include <MNN_generated.h>

namespace MNN {
namespace Train {
namespace Model {

// Multi-layer Perceptron used in Swin Transformer blocks
class MNN_PUBLIC Mlp : public Express::Module {
public:
    Mlp(int inFeatures, int hiddenFeatures = -1, int outFeatures = -1, float drop = 0.0f);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

private:
    std::shared_ptr<Express::Module> fc1;
    std::shared_ptr<Express::Module> fc2;
    std::shared_ptr<Express::Module> dropout;
};

// Window-based Multi-head Self Attention
class MNN_PUBLIC WindowAttention : public Express::Module {
public:
    WindowAttention(int dim, const std::vector<int>& windowSize, int numHeads, 
                   bool qkvBias = true, float qkScale = -1.0f, 
                   float attnDrop = 0.0f, float projDrop = 0.0f);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

private:
    int dim;
    std::vector<int> windowSize;
    int numHeads;
    float scale;
    std::shared_ptr<Express::Module> qkv;
    std::shared_ptr<Express::Module> attnDropout;
    std::shared_ptr<Express::Module> proj;
    std::shared_ptr<Express::Module> projDropout;
    Express::VARP relativePositionBiasTable;
    Express::VARP relativePositionIndex;
};

// Swin Transformer Block
class MNN_PUBLIC SwinTransformerBlock : public Express::Module {
public:
    SwinTransformerBlock(int dim, const std::vector<int>& inputResolution, int numHeads,
                        const std::vector<int>& windowSize = {7, 7}, 
                        const std::vector<int>& shiftSize = {0, 0},
                        float mlpRatio = 4.0f, bool qkvBias = true, float qkScale = -1.0f,
                        float drop = 0.0f, float attnDrop = 0.0f, float dropPath = 0.0f);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

private:
    int dim;
    std::vector<int> inputResolution;
    int numHeads;
    std::vector<int> windowSize;
    std::vector<int> shiftSize;
    float mlpRatio;
    std::shared_ptr<Express::Module> norm1;
    std::shared_ptr<WindowAttention> attn;
    std::shared_ptr<Express::Module> norm2;
    std::shared_ptr<Mlp> mlp;
    Express::VARP attnMask;
    float dropPathRate;
};

// Patch Merging Layer for downsampling
class MNN_PUBLIC PatchMerging : public Express::Module {
public:
    PatchMerging(const std::vector<int>& inputResolution, int dim);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

private:
    std::vector<int> inputResolution;
    int dim;
    std::shared_ptr<Express::Module> reduction;
    std::shared_ptr<Express::Module> norm;
};

// Basic Swin Transformer Layer
class MNN_PUBLIC BasicLayer : public Express::Module {
public:
    BasicLayer(int dim, const std::vector<int>& inputResolution, int depth, int numHeads,
              const std::vector<int>& windowSize, float mlpRatio = 4.0f,
              bool qkvBias = true, float qkScale = -1.0f, float drop = 0.0f,
              float attnDrop = 0.0f, const std::vector<float>& dropPath = {},
              bool useDownsample = true);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

private:
    int dim;
    std::vector<int> inputResolution;
    int depth;
    std::vector<std::shared_ptr<SwinTransformerBlock>> blocks;
    std::shared_ptr<PatchMerging> downsample;
};

// Patch Embedding Layer
class MNN_PUBLIC PatchEmbed : public Express::Module {
public:
    PatchEmbed(const std::vector<int>& imgSize = {224, 224}, 
              const std::vector<int>& patchSize = {4, 4},
              int inChannels = 3, int embedDim = 96);
    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;
    
    std::vector<int> getImgSize() const { return imgSize; }
    std::vector<int> getPatchSize() const { return patchSize; }
    std::vector<int> getPatchesResolution() const { return patchesResolution; }
    int getNumPatches() const { return numPatches; }

private:
    std::vector<int> imgSize;
    std::vector<int> patchSize;
    std::vector<int> patchesResolution;
    int numPatches;
    int inChannels;
    int embedDim;
    std::shared_ptr<Express::Module> proj;
    std::shared_ptr<Express::Module> norm;
};

// Multi-modal Swin Transformer for Sensor Data
class MNN_PUBLIC SwinTransformerV4_CMC : public Express::Module {
public:
    SwinTransformerV4_CMC(int numClasses = 7,
                            int numSegments = 10,
                            const std::vector<std::string>& modalities = {"seismic", "audio"},
                            const std::vector<std::string>& locations = {"shake"},
                            float dropRate = 0.2f, float dropPathRate = 0.1f, float attnDropRate = 0.2f,
                            int timeFreqOutChannels = 64, int timeFreqHeadNum = 4,
                            const std::map<std::string, std::vector<int>>& timeFreqBlockNum = {},
                            int locOutChannels = 256, float mlpRatio = 4.0f, bool qkvBias = true,
                            bool apeEnabled = false, const std::map<std::string, std::vector<int>>& windowSize = {},
                            const std::map<std::string, std::vector<int>>& patchSizeFreq = {},
                            const std::map<std::string, std::map<std::string, int>>& locModInFreqChannels = {},
                            const std::map<std::string, std::map<std::string, int>>& locModSpectrumLen = {},
                            const std::map<std::string, int>& inStride = {},
                            int fcDim = 512, const std::string& pretrainedHead = "linear");

    virtual std::vector<Express::VARP> onForward(const std::vector<Express::VARP> &inputs) override;

    // Forward methods
    std::vector<Express::VARP> forwardEncoder(const std::map<std::string, std::map<std::string, Express::VARP>>& patchedInputs);
    std::map<std::string, std::map<std::string, Express::VARP>> patchForward(const std::map<std::string, std::map<std::string, Express::VARP>>& freqX);
    std::pair<Express::VARP, std::vector<int>> padInput(Express::VARP freqInput, const std::string& loc, const std::string& mod);
    void initEncoder();

private:
    int numClasses;
    int numSegments;
    std::vector<std::string> modalities;
    std::vector<std::string> locations;
    
    // Configuration parameters
    float dropRate;
    float dropPathRate;
    float attnDropRate;
    int timeFreqOutChannels;
    int timeFreqHeadNum;
    std::map<std::string, std::vector<int>> timeFreqBlockNum;
    int locOutChannels;
    float mlpRatio;
    bool qkvBias;
    bool apeEnabled;
    std::map<std::string, std::vector<int>> windowSize;
    std::map<std::string, std::vector<int>> patchSizeFreq;
    std::map<std::string, std::map<std::string, int>> locModInFreqChannels;
    std::map<std::string, std::map<std::string, int>> locModSpectrumLen;
    std::map<std::string, int> inStride;
    int fcDim;
    std::string pretrainedHead;
    
    int sampleDim;
    
    // Network components
    std::map<std::string, std::map<std::string, std::vector<std::shared_ptr<BasicLayer>>>> freqIntervalLayers;
    std::map<std::string, std::map<std::string, std::shared_ptr<PatchEmbed>>> patchEmbed;
    std::map<std::string, std::map<std::string, Express::VARP>> absolutePosEmbed;
    std::map<std::string, std::map<std::string, std::shared_ptr<Express::Module>>> modInLayers;
    std::map<std::string, std::map<std::string, std::vector<int>>> imgSizes;
    
    // Classification layers
    std::shared_ptr<Express::Module> classLayer;
    std::shared_ptr<Express::Module> detectionClassLayer;
    
    std::shared_ptr<Express::Module> normLayer;
};

} // namespace Model
} // namespace Train
} // namespace MNN

#endif // SwinTransformer_hpp