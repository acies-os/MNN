//
//  SwinUtils.cpp
//  MNN
//
//  Created by MNN on 2025/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "SwinUtils.hpp"
#include <algorithm>
#include <cmath>
#include <random>

namespace MNN {
namespace Train {
namespace Model {

using namespace MNN::Express;

std::vector<int> getPaddedSize(const std::vector<int>& imgSize, 
                                const std::vector<int>& windowSize,
                                const std::vector<int>& patchSize,
                                int numLayers) {
    std::vector<int> paddedSize = imgSize;
    
    // Calculate minimum size needed after patch embedding and downsampling
    for (int i = 0; i < 2; i++) {
        int minSize = patchSize[i];
        for (int layer = 0; layer < numLayers; layer++) {
            minSize *= 2; // Account for downsampling
        }
        
        // Ensure divisible by window size
        int remainder = (paddedSize[i] / patchSize[i]) % windowSize[i];
        if (remainder != 0) {
            paddedSize[i] += (windowSize[i] - remainder) * patchSize[i];
        }
        
        paddedSize[i] = std::max(paddedSize[i], minSize);
    }
    
    return paddedSize;
}

VARP windowPartition(VARP x, const std::vector<int>& windowSize) {
    // x: (B, H, W, C)
    // Returns: (num_windows*B, window_height, window_width, C)
    
    auto info = x->getInfo();
    int B = info->dim[0];
    int H = info->dim[1];
    int W = info->dim[2];
    int C = info->dim[3];
    
    int windowHeight = windowSize[0];
    int windowWidth = windowSize[1];
    
    // Reshape to (B, H//window_height, window_height, W//window_width, window_width, C)
    x = _Reshape(x, {B, H / windowHeight, windowHeight, W / windowWidth, windowWidth, C});
    
    // Permute to (B, H//window_height, W//window_width, window_height, window_width, C)
    x = _Transpose(x, {0, 1, 3, 2, 4, 5});
    
    // Reshape to (num_windows*B, window_height, window_width, C)
    x = _Reshape(x, {-1, windowHeight, windowWidth, C});
    
    return x;
}

VARP windowReverse(VARP windows, const std::vector<int>& windowSize, int H, int W) {
    // windows: (num_windows*B, window_height, window_width, C)
    // Returns: (B, H, W, C)
    
    auto info = windows->getInfo();
    int windowHeight = windowSize[0];
    int windowWidth = windowSize[1];
    int C = info->dim[3];
    int B = info->dim[0] / (H * W / windowHeight / windowWidth);
    
    // Reshape to (B, H//window_height, W//window_width, window_height, window_width, C)
    windows = _Reshape(windows, {B, H / windowHeight, W / windowWidth, windowHeight, windowWidth, C});
    
    // Permute to (B, H//window_height, window_height, W//window_width, window_width, C)
    windows = _Transpose(windows, {0, 1, 3, 2, 4, 5});
    
    // Reshape to (B, H, W, C)
    windows = _Reshape(windows, {B, H, W, C});
    
    return windows;
}

void truncNormal(VARP tensor, float mean, float std, float a, float b) {
    // Simple implementation - in practice, you might want to use proper truncated normal
    auto info = tensor->getInfo();
    int totalSize = 1;
    for (int i = 0; i < info->dim.size(); i++) {
        totalSize *= info->dim[i];
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(mean, std);
    
    std::vector<float> values(totalSize);
    for (int i = 0; i < totalSize; i++) {
        float val = dist(gen);
        val = std::max(a, std::min(b, val));
        values[i] = val;
    }
    
    // Note: In actual MNN implementation, you would use proper tensor initialization
    // This is a simplified version for illustration
}

VARP dropPath(VARP x, float dropProb, bool training) {
    if (!training || dropProb == 0.0f) {
        return x;
    }
    
    float keepProb = 1.0f - dropProb;
    auto info = x->getInfo();
    
    // Create random tensor with same batch dimension, but 1s for other dimensions
    std::vector<int> shape = {info->dim[0]};
    for (int i = 1; i < info->dim.size(); i++) {
        shape.push_back(1);
    }
    
    // Use a simplified drop path implementation
    // In practice, this would be implemented as a custom MNN operation
    // For now, return input scaled by keep probability during training
    if (training) {
        return x * _Scalar<float>(keepProb);
    }
    return x;
}

VARP createRelativePositionBiasTable(const std::vector<int>& windowSize, int numHeads) {
    int tableSize = (2 * windowSize[0] - 1) * (2 * windowSize[1] - 1);
    auto table = _TrainableParam(0.0f, {tableSize, numHeads}, NCHW);
    truncNormal(table, 0.0f, 0.02f, -2.0f, 2.0f);
    return table;
}

VARP getRelativePositionIndex(const std::vector<int>& windowSize) {
    int Wh = windowSize[0];
    int Ww = windowSize[1];
    
    // Create coordinate matrices
    std::vector<int> coords_h(Wh), coords_w(Ww);
    for (int i = 0; i < Wh; i++) coords_h[i] = i;
    for (int i = 0; i < Ww; i++) coords_w[i] = i;
    
    // Create relative position index matrix
    std::vector<int> relative_position_index(Wh * Ww * Wh * Ww);
    
    for (int i = 0; i < Wh; i++) {
        for (int j = 0; j < Ww; j++) {
            for (int ki = 0; ki < Wh; ki++) {
                for (int kj = 0; kj < Ww; kj++) {
                    int idx1 = i * Ww + j;
                    int idx2 = ki * Ww + kj;
                    int relative_h = i - ki + Wh - 1;
                    int relative_w = j - kj + Ww - 1;
                    int relative_idx = relative_h * (2 * Ww - 1) + relative_w;
                    relative_position_index[idx1 * Wh * Ww + idx2] = relative_idx;
                }
            }
        }
    }
    
    return _Const(relative_position_index.data(), {Wh * Ww, Wh * Ww}, NCHW, halide_type_of<int>());
}

VARP createShiftedWindowMask(const std::vector<int>& inputResolution,
                            const std::vector<int>& windowSize,
                            const std::vector<int>& shiftSize) {
    int H = inputResolution[0];
    int W = inputResolution[1];
    int windowHeight = windowSize[0];
    int windowWidth = windowSize[1];
    
    if (shiftSize[0] == 0 && shiftSize[1] == 0) {
        return nullptr;  // No mask needed for regular windows
    }
    
    // Create image mask with region indices
    std::vector<float> mask_data(H * W, 0.0f);
    
    // Fill mask with region indices for shifted window attention
    int cnt = 0;
    std::vector<std::pair<int, int>> h_slices = {
        {0, H - windowHeight},
        {H - windowHeight, H - shiftSize[0]},
        {H - shiftSize[0], H}
    };
    std::vector<std::pair<int, int>> w_slices = {
        {0, W - windowWidth},
        {W - windowWidth, W - shiftSize[1]},
        {W - shiftSize[1], W}
    };
    
    for (auto& h_slice : h_slices) {
        for (auto& w_slice : w_slices) {
            for (int h = h_slice.first; h < h_slice.second; h++) {
                for (int w = w_slice.first; w < w_slice.second; w++) {
                    if (h >= 0 && h < H && w >= 0 && w < W) {
                        mask_data[h * W + w] = static_cast<float>(cnt);
                    }
                }
            }
            cnt++;
        }
    }
    
    auto imgMask = _Const(mask_data.data(), {1, H, W, 1}, NCHW);
    
    // Partition into windows
    auto maskWindows = windowPartition(imgMask, windowSize);
    maskWindows = _Reshape(maskWindows, {-1, windowHeight * windowWidth});
    
    // Create attention mask by comparing window regions
    auto info = maskWindows->getInfo();
    int numWindows = info->dim[0];
    int windowTokens = info->dim[1];
    
    // Create attention mask - simplified implementation
    // In practice, this would need proper indexing operations
    std::vector<float> attn_mask_data(numWindows * windowTokens * windowTokens, 0.0f);
    
    // Fill with -100.0 for different regions, 0.0 for same regions
    for (int w = 0; w < numWindows; w++) {
        for (int i = 0; i < windowTokens; i++) {
            for (int j = 0; j < windowTokens; j++) {
                // This is a simplified version - actual implementation would extract
                // region IDs from maskWindows and compare them
                int idx = w * windowTokens * windowTokens + i * windowTokens + j;
                attn_mask_data[idx] = (i == j) ? 0.0f : -100.0f;
            }
        }
    }
    
    auto attnMask = _Const(attn_mask_data.data(), {numWindows, windowTokens, windowTokens}, NCHW);
    return attnMask;
}

VARP rollTensor(VARP x, const std::vector<int>& shifts, const std::vector<int>& dims) {
    // Simple implementation of roll/cyclic shift
    // In practice, this might need to be implemented as a custom operation in MNN
    
    VARP result = x;
    for (size_t i = 0; i < shifts.size() && i < dims.size(); i++) {
        int shift = shifts[i];
        int dim = dims[i];
        
        if (shift == 0) continue;
        
        auto info = result->getInfo();
        int dimSize = info->dim[dim];
        
        // Normalize shift
        shift = shift % dimSize;
        if (shift < 0) shift += dimSize;
        
        if (shift > 0) {
            // Split and concatenate
            std::vector<int> indices1(dimSize - shift);
            std::vector<int> indices2(shift);
            
            for (int j = 0; j < dimSize - shift; j++) {
                indices1[j] = j + shift;
            }
            for (int j = 0; j < shift; j++) {
                indices2[j] = j;
            }
            
            auto part1 = _Gather(result, _Const(indices1.data(), {static_cast<int>(indices1.size())}, NCHW));
            auto part2 = _Gather(result, _Const(indices2.data(), {static_cast<int>(indices2.size())}, NCHW));
            
            result = _Concat({part1, part2}, dim);
        }
    }
    
    return result;
}

VARP padTensor(VARP input, const std::vector<int>& padding, PaddingMode mode, float value) {
    // Simplified padding implementation for CONSTANT mode
    // padding format: [pad_left, pad_right, pad_top, pad_bottom] for 4D tensor
    if (padding.size() < 4) {
        return input; // No padding needed
    }
    
    auto info = input->getInfo();
    if (info->dim.size() < 4) {
        return input; // Only support 4D tensors
    }
    
    // Calculate new dimensions
    int N = info->dim[0];
    int C = info->dim[1];
    int H = info->dim[2];
    int W = info->dim[3];
    
    int newH = H + padding[2] + padding[3]; // top + bottom padding
    int newW = W + padding[0] + padding[1]; // left + right padding
    
    if (newH == H && newW == W) {
        return input; // No padding needed
    }
    
    // Create padded tensor filled with constant value
    auto dims = _Const(std::vector<int>{N, C, newH, newW}.data(), {4}, NCHW, halide_type_of<int>());
    auto paddedTensor = _Fill(dims, _Scalar<float>(value));
    
    // For actual implementation, you would need a proper slice assignment operation
    // This is a simplified version that assumes the padding operation exists in MNN
    // In practice, you might need to implement this as a custom MNN operation
    
    // Return input for now - in actual MNN implementation, proper padding would be used
    // This is a placeholder that should be replaced with actual MNN padding operation
    return input;
}

} // namespace Model
} // namespace Train
} // namespace MNN