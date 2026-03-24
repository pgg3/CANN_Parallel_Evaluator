#ifndef RELU_CUSTOM_TILING_H
#define RELU_CUSTOM_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ReluCustomTilingData)
    TILING_DATA_FIELD_DEF(uint32_t, totalLength);
    TILING_DATA_FIELD_DEF(uint32_t, tileNum);
    TILING_DATA_FIELD_DEF(uint32_t, tileLength);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ReluCustom, ReluCustomTilingData)
}

#endif // RELU_CUSTOM_TILING_H
