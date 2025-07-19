//
//  SwinUtils.hpp
//  MNN
//
//  Created by MNN on 2025/01/08.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef SwinUtils_hpp
#define SwinUtils_hpp

#include <vector>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/Module.hpp>

#include "Initializer.hpp"
#include <vector>
#include <MNN/expr/Module.hpp>
#include "NN.hpp"
#include <algorithm>

namespace MNN {
namespace Train {
namespace Model {

// Helper function to calculate padded size for window partitioning
std::vector<int> getPaddedSize(const std::vector<int>& imgSize, 
                              const std::vector<int>& windowSize,
                              const std::vector<int>& patchSize,
                              int numLayers);

// Window partition function - partitions input into windows
Express::VARP windowPartition(Express::VARP x, const std::vector<int>& windowSize);

// Window reverse function - reverses window partitioning
Express::VARP windowReverse(Express::VARP windows, const std::vector<int>& windowSize, 
                           int H, int W);

// Truncated normal initialization
void truncNormal(Express::VARP tensor, float mean = 0.0f, float std = 1.0f, 
                float a = -2.0f, float b = 2.0f);

// Drop path implementation
Express::VARP dropPath(Express::VARP x, float dropProb, bool training = true);

// Relative position bias table initialization
Express::VARP createRelativePositionBiasTable(const std::vector<int>& windowSize, int numHeads);

// Get relative position index for window attention
Express::VARP getRelativePositionIndex(const std::vector<int>& windowSize);

// Create attention mask for shifted window attention
Express::VARP createShiftedWindowMask(const std::vector<int>& inputResolution,
                                     const std::vector<int>& windowSize,
                                     const std::vector<int>& shiftSize);

// Roll function for cyclic shifting (equivalent to torch.roll)
Express::VARP rollTensor(Express::VARP x, const std::vector<int>& shifts, const std::vector<int>& dims);

// Padding modes enumeration
enum class PaddingMode {
    CONSTANT = 0,
    REFLECT = 1,
    REPLICATE = 2
};

// Custom padding function
Express::VARP padTensor(Express::VARP input, const std::vector<int>& padding, PaddingMode mode = PaddingMode::CONSTANT, float value = 0.0f);

} // namespace Model
} // namespace Train
} // namespace MNN

#endif // SwinUtils_hpp