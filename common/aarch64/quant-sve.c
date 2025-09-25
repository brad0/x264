/*****************************************************************************
 * quant-sve.S: aarch64 trellis
 *****************************************************************************
 * Copyright (C) 2009-2025 x264 project
 *
 * Authors: Matthias Langer <mlanger@nvidia.com>,
 *          Alexander Komarov <akomarov@nvidia.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02111, USA.
 *
 * This program is also available under a commercial proprietary license.
 * For more information, contact us at licensing@x264.com.
 *****************************************************************************/

#include "common/common.h"
#include "common/bitstream.h"
#include <stdbool.h>

#define x264_cabac_size_unary x264_template(cabac_size_unary)
extern uint16_t x264_cabac_size_unary[15][128];

#define SIGN32(x,y) (((x)^((y) >> 31))-((y) >> 31))

extern int phase1_compute(const int last_nnz, const int b_ac, const dctcoef *restrict const quant_coefs, uint32_t *cur_level0123,
        uint32_t **restrict next_level, uint32_t *lev_used0123, const uint8_t *restrict const zigzag,
        const int *restrict const unquant_mf, const int dc, const dctcoef *restrict const orig_coefs, const int i_psy_trellis,
        const dctcoef *const fenc_dct, const uint32_t *restrict const coef_weight2, const uint32_t *restrict const coef_weight1,
        uint64_t *cur_score02, uint64_t *cur_score13, uint64_t *cur_score46, uint32_t *cur_cabac3, uint32_t *entropy012_,
        const int num_coefs, const int b_interlaced, const uint8_t *restrict const cabac_state_sig,
        const uint8_t *restrict const cabac_state_last, const uint64_t lambda2,
        uint32_t *entropy012_xor, const uint32_t level_state0489, uint32_t *cur_cabac4, uint32_t *cur_level4567,
        const uint16_t *restrict const cabac_size_unary_5, const uint64_t level_state01234567, int *phase2_compute_needed);

extern uint32_t phase2_compute(int i, const int b_ac, const dctcoef *restrict const quant_coefs, uint32_t *cur_level0123,
        uint32_t **restrict next_level, uint32_t *lev_used0123, const uint8_t *restrict const zigzag,
        const int *restrict const unquant_mf, const int dc, const dctcoef *restrict const orig_coefs,
        const int i_psy_trellis, const dctcoef *const fenc_dct, const uint32_t *restrict const coef_weight2,
        const uint32_t *restrict const coef_weight1, uint64_t *cur_score02, uint64_t *cur_score13, uint64_t *cur_score46,
        uint32_t *cur_cabac3, uint32_t *entropy012_, const int num_coefs, const int b_interlaced,
        const uint8_t *restrict const cabac_state_sig,const uint8_t *restrict const cabac_state_last, const uint64_t lambda2,
        uint32_t *entropy012_xor,const uint32_t level_state0489,uint32_t *cur_cabac4, uint32_t *cur_level4567,
        const uint16_t *restrict const cabac_size_unary_5, const uint64_t level_state01234567, const int switched, int *leveli, dctcoef *dcti, const int last_nnz, uint32_t *level_tree);

int trellis_cabac_sve(
    const int *restrict const unquant_mf, const uint8_t *restrict const zigzag,
    const uint64_t lambda2, const int last_nnz,
    const dctcoef *restrict const orig_coefs, const dctcoef *restrict const quant_coefs,
    dctcoef *restrict const dct,
    const uint8_t *restrict const cabac_state_sig, const uint8_t *restrict const cabac_state_last,
    const uint64_t level_state01234567, const uint16_t level_state89,
    const int num_coefs, const int dc, const int b_ac, const dctcoef *const fenc_dct,
    const int i_psy_trellis, const int b_interlaced
) {
    // (# of coefs) * (# of ctx) * (# of levels tried) = 1024
    // we don't need to keep all of those: (# of coefs) * (# of ctx) would be enough,
    // but it takes more time to remove dead states than you gain in reduced memory.
    ALIGNED_ARRAY_32( uint32_t, level_tree, [64 * 8 * 2] );
    memset(level_tree, 0, 64*16);
    uint32_t *restrict next_level = &level_tree[8];

    const uint32_t level_state0489 = (uint32_t)(
        (level_state01234567 & UINT64_C(0xff)) |
        ((level_state01234567 >> 24) & UINT64_C(0xff00)) |
        (level_state89 << 16));

    const uint32_t *restrict const coef_weight1 = num_coefs == 64 ? x264_dct8_weight_tab : x264_dct4_weight_tab;
    const uint32_t *restrict const coef_weight2 = num_coefs == 64 ? x264_dct8_weight2_tab : x264_dct4_weight2_tab;

    uint32_t cur_cabac3 = level_state0489, cur_cabac4 = 0;  // just contexts 0,4,8,9 of the 10 relevant to coding.

    /* levelgt1_ctx = [5, 5, 5, 5] */
    const uint16_t *restrict const cabac_size_unary_5 = &x264_cabac_size_unary[0][(level_state01234567 >> 40) & 0xff];

    // keep 2 versions of the main quantization loop, depending on which subsets of the node_ctxs are live
    // node_ctx 0..3, i.e. having not yet encountered any coefs that might be quantized to >1
    uint64_t cur_score02[2], cur_score13[2], cur_score46[2];
    uint32_t entropy012_[4], entropy012_xor[4], lev_used0123[4], cur_level0123[4], cur_level4567[4];
    int phase2_compute_needed = 0;
    int i = phase1_compute(last_nnz, b_ac, quant_coefs, cur_level0123,
        &next_level, lev_used0123, zigzag, unquant_mf, dc, orig_coefs, i_psy_trellis,
        fenc_dct, coef_weight2, coef_weight1, cur_score02, cur_score13, cur_score46,
        &cur_cabac3, entropy012_, num_coefs, b_interlaced, cabac_state_sig,
        cabac_state_last, lambda2, entropy012_xor, level_state0489, &cur_cabac4, cur_level4567,
        cabac_size_unary_5, level_state01234567, &phase2_compute_needed);

    uint32_t level;
    if (!phase2_compute(i, b_ac, quant_coefs, cur_level0123, &next_level, lev_used0123, zigzag,
         unquant_mf, dc, orig_coefs, i_psy_trellis, fenc_dct, coef_weight2, coef_weight1,
         cur_score02, cur_score13, cur_score46,&cur_cabac3, entropy012_, num_coefs,
         b_interlaced, cabac_state_sig,cabac_state_last, lambda2, entropy012_xor,level_state0489, &cur_cabac4, cur_level4567,
         cabac_size_unary_5, level_state01234567, phase2_compute_needed, &level, dct, last_nnz, level_tree))
    {
        return 0;
    }

    // This section is reached if:
    // 1. Phase 1 completed and the cur_score condition for 'return 0' was false.
    // 2. Phase 2 (and its preceding Phase 1 part) completed.
    // have 8 active nodes
    //for(i = b_ac; ; i + 3 <= last_nnz; i += 4 ) {}
    i = (b_ac + 3 > last_nnz) ? b_ac : (last_nnz + 1 - ((last_nnz - 3 - b_ac) % 4));
    for( ; i <= last_nnz; i++ )
    {
        const uint32_t leaf = level_tree[level];
        const uint32_t abs_lev = leaf >> 16;
        dct[zigzag[i]] = SIGN32(abs_lev, quant_coefs[i]); // equivalent to dct[zigzag[i]] );
        level = leaf & 0xffff;
    }
    return 1;
}

#define TRELLIS_ARGS unquant_mf, zigzag, lambda2, last_nnz, coefs, quant_coefs, dct,\
                     cabac_state_sig, cabac_state_last, level_state0, level_state1,\
                     i_coefs, dc, b_ac, fenc_dct, i_psy_trellis, b_interlaced

#define x264_trellis_cabac_4x4_sve x264_template(trellis_cabac_4x4_sve)
int x264_trellis_cabac_4x4_sve( TRELLIS_PARAMS, int b_ac ) {
    const int i_coefs = 16;
    const dctcoef *fenc_dct = NULL;
    const int i_psy_trellis = 0;
    const int b_interlaced = 0;
    const int dc = 0;
    return trellis_cabac_sve(TRELLIS_ARGS);
}

#define x264_trellis_cabac_4x4_psy_sve x264_template(trellis_cabac_4x4_psy_sve)
int x264_trellis_cabac_4x4_psy_sve ( TRELLIS_PARAMS, int b_ac, dctcoef *fenc_dct, int i_psy_trellis ) {
    const int i_coefs = 16;
    const int dc = 0;
    const int b_interlaced = 0;
    return trellis_cabac_sve(TRELLIS_ARGS);
}

#define x264_trellis_cabac_8x8_sve x264_template(trellis_cabac_8x8_sve)
int x264_trellis_cabac_8x8_sve( TRELLIS_PARAMS, int b_interlaced ) {
    const int i_coefs = 64;
    const int dc = 0;
    const int b_ac = 0;
    const dctcoef *fenc_dct = NULL;
    const int i_psy_trellis = 0;
    return trellis_cabac_sve(TRELLIS_ARGS);
}

#define x264_trellis_cabac_8x8_psy_sve x264_template(trellis_cabac_8x8_psy_sve)
int x264_trellis_cabac_8x8_psy_sve ( TRELLIS_PARAMS, int b_interlaced, dctcoef *fenc_dct, int i_psy_trellis ) {
    const int i_coefs = 64;
    const int dc = 0;
    const int b_ac = 0;
    return trellis_cabac_sve(TRELLIS_ARGS);
}

#define x264_trellis_cabac_chroma_422_dc_sve x264_template(trellis_cabac_chroma_422_dc_sve)
int x264_trellis_cabac_chroma_422_dc_sve ( TRELLIS_PARAMS ) {
    const int i_coefs = 8;
    const int dc = 1;
    const int b_ac = 0;
    const dctcoef *fenc_dct = NULL;
    const int i_psy_trellis = 0;
    const int b_interlaced = 0;
    return trellis_cabac_sve(TRELLIS_ARGS);
}

#define x264_trellis_cabac_dc_sve x264_template(trellis_cabac_dc_sve)
int x264_trellis_cabac_dc_sve ( TRELLIS_PARAMS, int i_coefs ) {
    const int b_ac = 0;
    const int dc = 1;
    const dctcoef *fenc_dct = NULL;
    const int i_psy_trellis = 0;
    const int b_interlaced = 0;
    return trellis_cabac_sve(TRELLIS_ARGS);
}

