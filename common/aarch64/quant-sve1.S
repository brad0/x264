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

#include "asm.S"
#define x264_glue3_expand(x,y,z) x##_##y##_##z
#define x264_glue3(x,y,z) x264_glue3_expand(x,y,z)
#define x264_template(w) x264_glue3(x264, BIT_DEPTH, w)
#define x264_cabac_transition_unary x264_template(cabac_transition_unary)
#define x264_cabac_size_unary x264_template(cabac_size_unary)
#define TRELLIS_SCORE_BIAS (1ULL<<60) // bias so that all valid scores are positive, even after negative contributions from psy

ENABLE_SVE
ENABLE_SVE2

const X(ue_size_tab)
    .byte   1,1,3,3,5,5,5,5,7,7,7,7,7,7,7,7
    .byte   9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9
    .byte   11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11
    .byte   11,11,11,11,11,11,11,11,11,11,11,11,11,11,11,11
    .byte   13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13
    .byte   13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13
    .byte   13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13
    .byte   13,13,13,13,13,13,13,13,13,13,13,13,13,13,13,13
    .byte   15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15
    .byte   15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15
    .byte   15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15
    .byte   15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15
    .byte   15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15
    .byte   15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15
    .byte   15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15
    .byte   15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15
endconst

.equ CABAC_SHIFT_CONST_VAL, 4 // LAMBDA_BITS 4, CABAC_SIZE_BITS 8 -> 8-4=4
#define CABAC_SIZE_BITS 8
#define LAMBDA_BITS     4

#define CUR_LEVEL0123_PTR  X3
#define LEV_USED0123_PTR   X5
#define CUR_SCORE02_PTR    X25
#define varW_LAMBDA2   W27
#define varX_LAMBDA2   X27

#define PREV_SCORE02   Z11
#define PREV_SCORE13   Z12
#define CUR_SCORE02    Z5
#define CUR_CABAC4567  Z6
#define CUR_SCORE13    Z13
#define SCORE11        Z25
#define TMP_Z0         Z26
#define TMP_Z1         Z27
#define TMP_Z2         Z28
#define TMP_Z3         Z24
#define TMP_Z5         Z22
#define LEV_USED0123   Z23
#define SCORE02        Z29
#define SCORE13        Z30
#define TMP0123        Z31
#define CUR_LEVEL0123  Z0
#define SSD_BOT        Z2.D
#define SSD_TOP        Z1.D

#define P_TRUE         P0
#define P_S0           P1
#define P_S2           P2
#define P_FALSE        P3
#define P_S3           P4
#define P_S23          P5
#define TMP_P1         P8
#define TMP_P2         P9
#define TMP_P3         P10
#define TMP_P4         P11
#define TMP_P5         P12
#define TMP_P6         P13
#define TMP_P7         P14

#define TMP_W2         W8
#define TMP_W3         W18
#define TMP_W6         W23

.macro  bs_size_ue_big_macro abs_level:req
    //--------------------------------------------------------------------
    // Macro: bs_size_ue_big_macro
    // Re-implements the static inline C function.
    // Input:   W18 - The value 'val' to be encoded.
    //--------------------------------------------------------------------
    movrel x22, X(ue_size_tab)
    sub TMP_W6, \abs_level, 15

    cmp     \abs_level, #255
    b.ge    .L_val_is_large_\@

.L_val_is_small_\@:
    // Path for val < 255
    add     TMP_W6, TMP_W6, #1
    ldrb    TMP_W3, [x22, TMP_W6, uxtw]
    b       .L_val_lookup_done_\@

.L_val_is_large_\@:
    // Path for val >= 255
    add     TMP_W6, TMP_W6, #1
    lsr     TMP_W6, TMP_W6, #8
    ldrb    TMP_W3, [x22, TMP_W6, uxtw]
    add     TMP_W3, TMP_W3, #16

.L_val_lookup_done_\@:
    // Result is now in W18/TMP_W3
.endm

.macro TRELLIS_COEF1_0_SVE_ stage:req
    // ssd_bot = stage ? svzip2_u64( ssd0, ssd1 ) : svzip1_u64( ssd0, ssd1 );
    .if \stage
        zip2 SSD_BOT, z4.d, z7.d
    .else
        zip1 SSD_BOT, z4.d, z7.d
    .endif
    // ssd_top = svdup_lane_u64( ssd1, stage );
    dup SSD_TOP, z7.d[\stage]

    // tmp0123 = svadd_n_u32_x( p, cost_siglast_2111, 1 << CABAC_SIZE_BITS );
    // cost_siglast_2111 in z6
    movprfx TMP0123, z6
    add TMP0123.s, TMP0123.s, #(1 << CABAC_SIZE_BITS)
    // tmp0123 = svadd_u32_x( p, tmp0123, entropy012_ );
    // entropy012_ in in Z3
    add TMP0123.s, TMP0123.s, z3.s

    // score02 = svadd_u64_x( p, prev_score02, ssd_bot );
    // score13 = svadd_u64_x( p, prev_score13, ssd_top );
    mov SCORE02.d, PREV_SCORE02.d
    add SCORE02.d, SCORE02.d, SSD_BOT
    mov SCORE13.d, PREV_SCORE13.d
    add SCORE13.d, SCORE13.d, SSD_TOP

    // score02 = svsra_n_u64( score02, svmullb_n_u64(tmp0123, lambda2), CABAC_SIZE_BITS - LAMBDA_BITS );
    // score13 = svsra_n_u64( score13, svmullt_n_u64(tmp0123, lambda2), CABAC_SIZE_BITS - LAMBDA_BITS );
    // Need widening multiply u32 vector by u64 scalar -> u64 vector
    dup TMP_Z0.s, varW_LAMBDA2
    umullb TMP_Z1.d, TMP0123.s, TMP_Z0.s // Multiply by lambda2
    umullt TMP_Z2.d, TMP0123.s, TMP_Z0.s

    // Shift and Add (unsigned rounding shift right)
    usra SCORE02.d, TMP_Z1.d, #(CABAC_SIZE_BITS - LAMBDA_BITS) // Shift lower 64-bit results
    usra SCORE13.d, TMP_Z2.d, #(CABAC_SIZE_BITS - LAMBDA_BITS) // Shift upper 64-bit results
    // temp_z11.d = score02, temp_z12.d = score13

    // score11 = svdup_lane_u64( score13, 0 );
    mov SCORE11.d, SCORE13.d[0] // temp_z13.d = score11

    // Predicate calculations and score updates
    // prev_j_1 = svuzp1_b64( pf, prev_j13 );
    uzp1 p6.d, P_FALSE.d, p4.d // sve_temp1_p = prev_j_1

    // prev_j_3 = svbic_b_z( p, prev_j13, s0 );
    bic p4.b, P_TRUE/Z, p4.b, P_S0.b // sve_temp2_p = prev_j_3

    // j02 = prev_j02;
    mov p7.b, p5.b // j02 is correctly initialized with prev_j02
    .if \stage
        // j02 = svcmplt_u64( j02, score02, cur_score13 );
        cmphi p7.d, p5/z, CUR_SCORE13.d, SCORE02.d
    .endif
    // cur_score13 = svmin_u64_m( prev_j02, cur_score13, score02 );
    umin CUR_SCORE13.d, p5/m, CUR_SCORE13.d, SCORE02.d

    // j_1 = prev_j_1;
    mov p1.b, p6.b
    .if \stage
        // j_1 = svcmplt_u64( j_1, score11, cur_score02 );
        cmphi p1.d, p6/z, CUR_SCORE02.d, SCORE11.d
    .endif
    // cur_score02 = svmin_u64_m( prev_j_1, cur_score02, score11 );
    umin CUR_SCORE02.d, p6/m, CUR_SCORE02.d, SCORE11.d
    str CUR_SCORE02, [CUR_SCORE02_PTR]  // Store *cur_score02

    // j_3 = prev_j_3;
    mov p5.b, p4.b // p4.d = j_3
    // j_3 = svcmplt_u64( j_3, score13, cur_score13 );
    cmphi p5.d, p4/z, CUR_SCORE13.d, SCORE13.d
    // cur_score13 = svmin_u64_m( prev_j_3, cur_score13, score13 );
    umin CUR_SCORE13.d, p4/m, CUR_SCORE13.d, SCORE13.d

    ldr x13, [x29, #168]
    str CUR_SCORE13, [x13]

    // j_2not3 = svbic_b_z( s2, j02, j_3 );
    bic p4.b, p2/z, p7.b, p5.b

    // j_012 = svtrn1_b32( j_1, j02 );
    trn1 p6.s, p1.s, p7.s // j_012

    // j___3 = svuzp1_b32( pf, j_3 );
    uzp1 p7.s, P_FALSE.s, p5.s // j___3

    // cur_cabac3 = svptest_any(p, j_2not3) ? level_state0489 : cur_cabac3;
    // Test if any active element of j_2not3 (p5) is true, controlled by predicate p (sve_p_p) at 32-bit element size
    ptest P_TRUE, p4.b
    ldr x13, [x29, #184]           // X13 = ptr_cur_cabac3 (arg 18)
    ldr w15, [x29, #248]           // W15 now is level_state0489 (arg 26)
    ldr w14, [x13]                 // W14 (W_cur_cabac3_val)
    csel w14, w15, w14, ne // Select based on condition

    // cur_cabac3 ^= (cur_cabac3 ^ transition3) & (svptest_any(p, j_3) ? 0xff00 : 0);
    // Test if any active element of j_3 (p4) is true, controlled by predicate p (sve_p_p) at 32-bit element size
    eor TMP_W2, w14, w24
    ptest P_TRUE, p5.b
    mov w18, #0xff00                // Active mask is 0xff00
    csel w17, w18, wzr, ne          // Select mask based on PTEST result
    and TMP_W2, TMP_W2, w17         // Apply the mask
    eor w14, w14, TMP_W2            // Final XOR to update cur_cabac3
    str w14, [x13]                  // Store the final result back to memory

    // cur_level0123 = svsel_u32( j_012, svsub_n_u32_x( p, lev_used0123, 1 ), cur_level0123 );
    // j_012 in p6.s, lev_used0123 in lev_used0123_z_s, cur_level0123 in cur_level0123_z_s
    ldr TMP_Z1, [LEV_USED0123_PTR]
    ldr TMP_Z3, [CUR_LEVEL0123_PTR]
    mov LEV_USED0123.d, TMP_Z1.d
    sub TMP_Z1.s, TMP_Z1.s, #1
    sel CUR_LEVEL0123.s, p6, TMP_Z1.s, TMP_Z3.s
    sel CUR_LEVEL0123.s, p7, LEV_USED0123.s, CUR_LEVEL0123.s
    str CUR_LEVEL0123, [CUR_LEVEL0123_PTR]

    // lev_used0123 = svadd_n_u32_x( p, lev_used0123, 4 );
    add LEV_USED0123.s, LEV_USED0123.s, #4
    str LEV_USED0123, [LEV_USED0123_PTR]
.endm

.macro TRELLIS_COEF2_0_SVE_
    // const svuint64_t ssd_bot = svzip2_u64( ssd0, ssd1 );
    // Z4.D = ssd0 (input), Z7.D = ssd1 (input). Result Z_ssd_bot in Z2.D.
    zip2 SSD_BOT, z4.d, z7.d

    // const svuint64_t ssd_top = svdup_lane_u64( ssd1, 1);
    // Z0.D still holds original ssd1. Result Z_ssd_top in Z1.D.
    dup SSD_TOP, z7.d[1]

    // svuint32_t tmp0123 = svadd_u32_x( p, cost_siglast_2111, *entropy012_xor );
    ldr x13, [x29, #240]           // X13 = ptr_entropy012_xor
    ldr z0, [x13]   // Z0.S = *entropy012_xor (using Z0 as temp for loaded entropy)
    mov z3.d, z6.d
    add z3.s, z3.s, z0.s     // Z3.S (tmp0123) = cost_siglast_2111(Z7.S) + Z0.S

    // const svuint32_t size_unary0123 = svdup_u32( cabac_size_unary_5[128] );
    ldr x13, [x29, #272]           // X13 = ptr_cabac_size_unary_5
    add x17, x13, #(128 * 2)       // Offset for [128] (uint16_t elements)
    ldrh w14, [x17]                // W14 = cabac_size_unary_5[128] scalar value
    dup z0.s, w14                  // Z0.S (size_unary0123) = duplicated W14 (reusing Z0)

    // tmp0123 = svadd_u32_x( p, tmp0123, size_unary0123 );
    add z3.s, z3.s, z0.s     // Z3.S (tmp0123) += Z0.S

    // svuint64_t score02 = svadd_u64_x( p, prev_score02, ssd_bot );
    mov SCORE02.d, PREV_SCORE02.d
    add SCORE02.d, SCORE02.d, SSD_BOT     // (score02) = prev_score02 + ssd_bot

    // svuint64_t score13 = svadd_u64_x( p, prev_score13, ssd_top );
    mov SCORE13.d, PREV_SCORE13.d
    add SCORE13.d, SCORE13.d, SSD_TOP     // (score13) = prev_score13 + ssd_top

    // score02 = svsra_n_u64( score02, svmullb_n_u64(tmp0123, lambda2), CABAC_SIZE_BITS - LAMBDA_BITS );
    // Z3.S = tmp0123, W27 = lambda2. Shift = 4.
    mov z0.s, varW_LAMBDA2
    umullb z0.d, z0.s, z3.s        // Z0.D = mullb(Z3.S, W27) (SVE2 scalar variant)
    usra SCORE02.d, z0.d, #4       // (score02) += asr(Z0.D, #4)

    // score13 = svsra_n_u64( score13, svmullt_n_u64(tmp0123, lambda2), CABAC_SIZE_BITS - LAMBDA_BITS );
    mov    z1.s, varW_LAMBDA2
    umullt z1.d, z1.s, z3.s        // Z1.D = mullt(Z3.S, W27)
    usra   SCORE13.d, z1.d, #4     // (score13) += asr(Z1.D, #4)

    // const svuint64_t score22 = svdup_lane_u64( score02, 1 );
    dup z0.d, SCORE02.d[1]         // Z0.D (score22) = score02[1]

    // const svuint64_t score33 = svdup_lane_u64( score13, 1 );
    dup z1.d, SCORE13.d[1]         // Z1.D (score33) = score13[1]

    // Predicate logic preparation
    mov  p9.b, p4.b                // Save prev_j13_input into P9
    and  p4.b, P_S0/Z, p4.b, p9.b  // P4 = s0 && prev_j13_input (P_j1_for_cmplt)
    uzp2 p2.d, P5.d, P_FALSE.d     // P2 = uzp2(prev_j02_input(P2), pf) (P_j2_for_cmplt)

    // Load *cur_score46 into Z4.D
    ldr  x13, [x29, #176]           // X13 = ptr_cur_score46
    ldr z4, [x13]    // Z4.D = *cur_score46 (governed by original P0=p_true)
    uzp2  p7.d, TMP_P7.d, P_FALSE.d // P7 = uzp2(prev_j13_input, pf) (P_j3_for_cmplt)

    // j0 block: svbool_t j0 = s0; j0 = svcmplt_u64( j0, score02, *cur_score46 );
    // *cur_score46 = svmin_u64_m( s0, *cur_score46, score02 );
    // Governing for CMPLT is s0 (P1).
    cmphi p9.d, P_S0/Z, z4.d, SCORE02.d
    umin  z4.d, P_S0/M, z4.d, SCORE02.d    // *cur_score46 = min under s0_mask (P1)

    // j1 block: svbool_t j1 = P_j1_for_cmplt (P4); j1 = svcmplt_u64( j1, score13, *cur_score46 );
    // *cur_score46 = svmin_u64_m( P_j1_for_cmplt, *cur_score46, score13 );
    // Governing for CMPLT is P_j1_for_cmplt (P4).
    cmphi p9.d, p4/z, z4.d, SCORE13.d   // P4 (P_j1_final_cond) updated
    umin z4.d, p4/m, z4.d, SCORE13.d    // *cur_score46 = min under P_j1_for_cmplt (P3)

    // j2 block: svbool_t j2 = P_j2_for_cmplt (P2); j2 = svcmplt_u64( j2, score22, *cur_score46 );
    // *cur_score46 = svmin_u64_m( P_j2_for_cmplt, *cur_score46, score22 );
    // score22 in Z0.D. Governing for CMPLT is P_j2_for_cmplt (P2). Result P_j2_final in P2.
    mov   TMP_Z5.d, z4.d
    umin  z4.d, p2/m, z4.d, z0.d       // *cur_score46 = min under P_j2_for_cmplt (P2)
    cmphi p2.d, p2/z, TMP_Z5.d, z0.d   // P2 (P_j2_final_cond) updated

    // j3 block: svbool_t j3 = P_p0_temp_j3 (current P7); j3 = svcmplt_u64( j3, score33, *cur_score46 );
    // *cur_score46 = svmin_u64_m( P_p0_temp_j3, *cur_score46, score33 );
    // score33 in Z1.D.
    mov   TMP_Z5.d, z4.d
    umin  z4.d, p7/m, z4.d, z1.d       // *cur_score46 = min under P_p0_temp_j3 (current P0)
    cmphi p7.d, p7/z, TMP_Z5.d, z1.d   // P7 (P_j3_final_cond) updated

    // Store updated *cur_score46. Use original P0 (p_true) for governing store if available, else PTRUE a temp pred.
    str z4, [x13] // X13 still ptr_cur_score46

    // *cur_cabac4 = level_state0489;
    ldr w14, [x29, #248]           // W14 = level_state0489
    ldr x13, [x29, #256]           // X13 = ptr_cur_cabac4
    str w14, [x13]

    // *cur_cabac4 ^= (*cur_cabac4 ^ transition3) & (svptest_any(p, j3) ? 0xff00 : 0);
    // j3 is in P7 (P_j3_final_cond). Original p_true (governing for PTEST) is P_all_true_temp.
    ptest P_TRUE, p7.b    // Test P_j3_final_cond (in P7) under mask P_all_true_temp
    mov w17, #0xff00
    csel w15, w17, wzr, ne   // W15 = mask (0xff00 or 0)
    ldr w18, [x13]     // X13 is ptr_cur_cabac4
    eor w17, w18, w24  // W17 = *cur_cabac4 ^ transition3
    and w17, w17, w15
    eor w18, w18, w17
    str w18, [x13]

    // *cur_level4567 = *lev_used0123;
    ldr LEV_USED0123, [LEV_USED0123_PTR] // Z7.S = *lev_used0123
    ldr x13, [x29, #264]           // X13 = ptr_cur_level4567
    str LEV_USED0123, [x13]

    // *cur_level4567 = svsel_u32( j1, svdup_lane_u32(*lev_used0123, 1), *cur_level4567 );
    // j1 is in P4 (P_j1_final_cond).
    ldr z0, [x13]  // Z0.S = current *cur_level4567
    mov  z1.d, LEV_USED0123.d
    dup  z1.s, LEV_USED0123.s[1]  // Z1.S = dup_lane(*lev_used0123, 1)
    sel  z0.s, p9, z1.s, z0.s     // Select under j1 (P4)
    str z0, [x13]

    // *cur_level4567 = svsel_u32( j2, svdup_lane_u32(*lev_used0123, 2), *cur_level4567 );
    // j2 is in P2 (P_j2_final_cond).
    ldr z0, [x13]
    mov z1.d, LEV_USED0123.d
    dup z1.s, LEV_USED0123.s[2]
    sel z0.s, p2, z1.s, z0.s       // Select under j2 (P2)
    str z0, [x13]

    // *cur_level4567 = svsel_u32( j3, svdup_lane_u32(*lev_used0123, 3), *cur_level4567 );
    // j3 is in P0 (P_j3_final_cond).
    ldr z0, [x13]
    mov z1.d, LEV_USED0123.d
    dup z1.s, LEV_USED0123.s[3]
    sel z0.s, p7, z1.s, z0.s       // Select under j3 (P7)
    str z0, [x13]

    // *lev_used0123 = svadd_n_u32_x( p, *lev_used0123, 4 );
    // *lev_used0123 value is in Z7.S. p is P_all_true_temp.
    add LEV_USED0123.s, LEV_USED0123.s, #4
    str LEV_USED0123, [LEV_USED0123_PTR]    // Store back to *lev_used0123
.endm

.macro TRELLIS_COEFN_0_SVE_ stage:req abs_level:req
    // ssd_bot = stage ? svzip2(ssd0, ssd1) : svzip1(ssd0, ssd1);
    // ssd_top = svdup_lane(ssd1, stage);
    .if \stage
        zip2 SSD_BOT, z4.d, z7.d
    .else
        zip1 SSD_BOT, z4.d, z7.d
    .endif
    dup SSD_TOP, z7.d[\stage]

    // prefix = X264_MIN(abs_level-1, 14)
    sub  TMP_W2, \abs_level, #1
    mov  TMP_W3, #14
    cmp  TMP_W2, TMP_W3
    csel TMP_W2, TMP_W2, TMP_W3, le // tmp_w2 = prefix

    // suffix_cost = (abs_level >= 15) ? (bs_size_ue_big(abs_level-15) << CABAC_SIZE_BITS) : 0;
    mov  TMP_W3, wzr               // w20 = suffix_cost, init to 0
    cmp  \abs_level, #15
    blt  .L_SKIP_SUFFIX_CALC_\@
    // Inlined bs_size_ue_big(\abs_level - 15)

    bs_size_ue_big_macro \abs_level
    lsl  TMP_W3, TMP_W3, #CABAC_SIZE_BITS

.L_SKIP_SUFFIX_CALC_\@:
    // tmp0123 = svadd(cost_siglast_2111, suffix_cost);
    dup  z3.s, TMP_W3
    mov  z31.d, z6.d
    add  z31.s,  z31.s, z3.s

    // tmp0123 += *entropy012_xor;
    ldr x13, [x29, #240]           // X13 = ptr_entropy012_xor
    ldr z3, [x13]
    add z31.s, z31.s, z3.s

    // tmp0123 += svdup(cabac_size_unary_5[prefix * 128]);
    ubfiz  x8, x8, #8, #32
    ldr  x17, [x29, #272]
    add  x8, x17, TMP_W2, uxtw
    ldrh  TMP_W3, [x8]
    dup  z3.s, TMP_W3
    add  z31.s, z31.s, z3.s

    // score02 = prev_score02 + ssd_bot; score13 = prev_score13 + ssd_top;
    mov  SCORE02.d, PREV_SCORE02.d
    add  SCORE02.d, SCORE02.d, SSD_BOT
    mov  SCORE13.d, PREV_SCORE13.d
    add  SCORE13.d, SCORE13.d, SSD_TOP

    // score_vec = svsra(score_vec, svmull(tmp0123, lambda2), shift);
    dup  z3.s, varW_LAMBDA2
    umullb z2.d, z31.s, z3.s
    usra   SCORE02.d, z2.d, #(CABAC_SIZE_BITS - LAMBDA_BITS)
    umullt z2.d, z31.s, z3.s
    usra   SCORE13.d, z2.d, #(CABAC_SIZE_BITS - LAMBDA_BITS)

    dup  z28.d, SCORE02.d[1]
    dup  z27.d, SCORE13.d[1]

    // Predicate generation
    and  p2.b, P_TRUE/Z, P_S0.b, p4.b
    uzp2 p6.d, p5.d, P_FALSE.d
    uzp2 p7.d, P4.d, P_FALSE.d

    // Load current score
    ldr x10, [x29, #176]
    ldr z26, [x10]       // z26 = cur_score46

    // Comparisons and updates for j=0,1,2,3
    mov  p8.b, P_S0.b    // p8 = j0 = s0
    .if \stage
        cmphi p8.d, P_S0/Z, z26.d, SCORE02.d
    .endif
    umin z26.d, P_S0/M, z26.d, SCORE02.d

    cmphi p9.d, p2/z, z26.d, SCORE13.d
    umin  z26.d, p2/m, z26.d, SCORE13.d

    cmphi p10.d, p6/z, z26.d, z28.d
    umin  z26.d, p6/m, z26.d, z28.d

    cmphi p11.d, p7/z, z26.d, z27.d
    umin  z26.d, p7/m, z26.d, z27.d

    str z26, [x10]          // Store updated cur_score46

    // CABAC state update
    ldr  x10, [x29, #256]   // X10 = ptr_cur_cabac4
    ldr  TMP_W3, [x10]      // w18 = cur_cabac4
    ldr  w15, [x29, #248]   // X15 = ptr_cur_level4567
    .if \stage
        orr p12.b, P_TRUE/Z, p8.b, p9.b   // p12 = j0 | j1
        orr p12.b, P_TRUE/Z, p12.b, p10.b // p12 = j0|j1 | j2
        orr p12.b, P_TRUE/Z, p12.b, p11.b // p12 = j0|j1|j2 | j3
        ptest P_TRUE, p12.b
        csel TMP_W3, w15, TMP_W3, ne      // Select if any comparison was true
    .else
        mov  TMP_W3, w15
    .endif
    str  TMP_W3, [x10]

    // *cur_cabac4 ^= (*cur_cabac4 ^ transition3) & (svptest_any(p, j3) ? 0xff00 : 0);
    ptest P_TRUE, p11.b
    mov  w8, #0xff00
    csel w19, w8, wzr, ne
    eor  w8, TMP_W3, w24
    and  w8, w8, w19
    eor  TMP_W3, TMP_W3, w8
    str  TMP_W3, [x10] // Store updated cur_cabac4

    // Level index update
    ldr x11, [x29, #264]                 // x11 = ptr_cur_level4567
    ldr LEV_USED0123, [LEV_USED0123_PTR] // z22 = lev_used0123
    .if \stage
        ldr z24, [x11]     // z24 = cur_level4567
        sel z24.s, p8, LEV_USED0123.s, z24.s
    .else
        mov z24.d, LEV_USED0123.d
    .endif

    dup z3.s, LEV_USED0123.s[1]
    sel z24.s, p9, z3.s, z24.s
    dup z3.s, LEV_USED0123.s[2]
    sel z24.s, p10, z3.s, z24.s
    dup z3.s, LEV_USED0123.s[3]
    sel z24.s, p11, z3.s, z24.s
    str z24, [x11]          // Store updated cur_level4567

    // lev_used0123 += 4;
    add LEV_USED0123.s, LEV_USED0123.s, #4
    str LEV_USED0123, [LEV_USED0123_PTR]
.endm

#define var_DC         W19
function trellis_cabac_phase1, export=1
phase1_compute:
    // Minimal prologue, only called from trellis_cabac_sve
    stp x29, x30, [sp, #-112]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]

// Load stack arguments into callee-saved registers
// last_nnz              W0
// b_ac                  W1
// *quant_coefs          X2
// *cur_level0123        X3 -> Z0
// uint32_t **next_level X4
// zigzag                X6
// unquant_mf            X7
    ldr var_DC, [x29, #112]       // W19 = dc (arg 9)
    ldp x20, x21, [x29, #120]     // X20 = orig_coefs (arg 10), reuse soon // X21 = i_psy_trellis (arg 11), reuse soon
    ldp x22, x23, [x29, #136]     // X22 = fenc_dct (arg 12), reuse soon, X23 = coef_weight2 (arg 13), reuse soon
    ldp x24, CUR_SCORE02_PTR, [x29, #152]          // X24 = coef_weight1 (arg 14), reuse soon
    ldr w26, [x29, #200]          // W26 = num_coefs (arg 20)
    ldr w28, [x29, #208]          // W28 = b_interlaced (arg 21), reuse very soon
    ldr varX_LAMBDA2, [x29, #232] // X27 = lambda2 (arg 24)

    // Load addresses of global tables into X9-X12
    movrel x9, X264(significant_coeff_flag_offset_8x8)
    movrel x10, X264(cabac_entropy)
    movrel x11, X264(last_coeff_flag_offset_8x8)
    movrel x12, X264(cabac_transition)

    // Initial SVE Predicate Setup
    ptrue  P_TRUE.b, vl16
    pfalse P_FALSE.b

    index z0.s, #0, #1      // Z_idx = {0, 1, 2, 3}
    mov   w13, #2
    mov   z1.s, w13
    cmpeq p2.s, P_TRUE/Z, z0.s, z1.s // P2 = (Z_idx == 2)
    ptrue p4.s, vl3

    // W0 = last_nnz (loop counter 'i')
    mov w13, wzr         // W13 = switched_to_phase2 = 0
    str w13, [sp, #96]   // Save it to stack, will need in the end
    ldr x16, [x4]        // X16 = *next_level (address from X4 which is &next_level)
    dup CUR_SCORE02.d, #-1
    dup CUR_SCORE13.d, #-1
    mov x13, #TRELLIS_SCORE_BIAS
    insr CUR_SCORE02.d, x13
    ldr x13, [x29, #168] // cur_score13
    str CUR_SCORE13, [x13]
    str CUR_SCORE02, [CUR_SCORE02_PTR]
    ldr x13, [x29, #176] // cur_score46
    str CUR_SCORE13, [x13]

    // entropy012_ = svld1uh_gather_u32index_u32(s012, cabac_entropy, level_state1234)
    ldr x13, [x29, #280] // level_state01234567
    lsr x13, x13, #8
    dup z0.d, x13
    uunpklo z1.h, z0.b
    uunpklo z2.s, z1.h
    ld1h z3.s, p4/z, [x10, z2.s, uxtw #1]
    // entropy012_xor
    eor  z2.s, z2.s,  #1
    ld1h z4.s, p4/z, [x10, z2.s, uxtw #1]

    add x13, x29, #192
    ldr x28, [x13]
    str z3, [x28]

    add x13, x29, #240
    ldr x28, [x13]
    str z4, [x28]

    index z3.s, #8, #1
    str z3, [x5] // lev_used0123 init

    dup z3.s, #0
    str z3, [x3] //cur_level0123

    ldr x13, [x29, #264]
    str z3, [x13] //cur_level0123

phase1_loop_condition:
    cmp w0, w1           // Compare i with b_ac
    b.lt phase1_loop_end_vl128
    ldr w28, [x29, #208] // reload b_interlaced
    ldr x22, [x29, #136] // reload clobbered
    ldr x20, [x29, #120] // X20 = orig_coefs

    mov w13, #1
    whilelo P_S0.s, wzr, w13       // P1 = 's0': {T,F,F,F}
    // sig_idx calculation
    mov w14, w0                    // W14 (W_sig_idx_calc) = i (W0)
    cmp var_DC, #0                 // if (dc (W19) == 0)
    b.ne dc_is_true_sig_idx_vl128
    cmp w26, #64                   // if (num_coefs (W26) == 64)
    b.ne not_64_coeffs_sig_idx_vl128
    // sig_idx = x264_significant_coeff_flag_offset_8x8[b_interlaced][i];
    mov w15, #64                   // W15 = 64 (coeffs_per_row_bytes if elements are uint8_t)
    mul w17, w28, w15              // W17 (W_row_offset) = b_interlaced (W28) * 64
    add x17, x9, w17, uxtw         // X17 (X_base_of_row) = X9 (table_base) + row_offset
    ldrb w14, [x17, w0, uxtw]      // W14 (W_sig_idx_calc) = table[row_base + i(W0)] (byte load)
    b sig_idx_done_vl128
dc_is_true_sig_idx_vl128:
    cmp w26, #8
    b.ne not_8_coeffs_dc_sig_idx_vl128
    lsr w14, w0, #1                // W14 (W_sig_idx_calc) = i (W0) / 2
    mov w15, #2                    // W15 (W_two_sig)
    cmp w14, w15
    csel w14, w14, w15, lt         // min(i/2, 2)
    b sig_idx_done_vl128
not_64_coeffs_sig_idx_vl128:
not_8_coeffs_dc_sig_idx_vl128:
sig_idx_done_vl128:
    // W14 now holds sig_idx
    ldr x15, [x29, #216]           // X15 = cabac_state_sig pointer (arg 22)
    ldrb w17, [x15, w14, uxtw]     // W17 (W_sig) = cabac_state_sig[sig_idx(W14)]
    mov w18, w17
    ldrh w15, [x10, w17, uxtw #1]  // W15 = x264_cabac_entropy[W_sig]
    dup TMP_Z0.d, x15              // cost_siglast_00_ssd

    dup z7.d, varX_LAMBDA2
    mul TMP_Z0.d, TMP_Z0.d, z7.d
    lsr TMP_Z0.d, TMP_Z0.d, #CABAC_SHIFT_CONST_VAL

    add x17, x2, w0, uxtw #1       // X17 = addr for quant_coefs[i]
    ldrsh x14, [x17]               // X14 = X_q

    cmp x14, #0                    // if (q == 0)
    b.ne q_is_not_zero_vl128

    ldr CUR_LEVEL0123, [CUR_LEVEL0123_PTR] // Load cur_level0123 (X3 address)
    str CUR_LEVEL0123, [x16]       // Store to *next_level (X4 is *next_level pointer)

    add x16, x16, #32              // *next_level += 32 bytes
    str x16, [x4]                  // Update *next_level in memory (X4 is &next_level)

    ldr CUR_SCORE02, [CUR_SCORE02_PTR]
    mov SCORE02.d, CUR_SCORE02.d
    sub CUR_SCORE02.d, P_S0/M, CUR_SCORE02.d, TMP_Z0.d
    str CUR_SCORE02, [CUR_SCORE02_PTR]

    ldr z1, [CUR_LEVEL0123_PTR]
    ldr LEV_USED0123, [LEV_USED0123_PTR]
    sel z1.s, P_S0, z1.s, LEV_USED0123.s
    str z1, [CUR_LEVEL0123_PTR]

    add LEV_USED0123.s, LEV_USED0123.s, #8
    str LEV_USED0123, [LEV_USED0123_PTR]

    b loop_continue_vl128

q_is_not_zero_vl128:
    cneg x28, x14, lt            // X28 (X_q_abs) = labs(q)

    dup z1.d, x28
    dup z7.d, #1
    sub z1.d, P_S0/M, z1.d, z7.d

    ldrb w15, [x6, w0, uxtw]     // W15 (W_zzi) = zigzag[i(W0)] (X6 base)
    uxtw x13, w15                // X13 (X_unquant_mf_idx) = zzi
    cmp var_DC, #0               // if (dc (W19) == 0)
    csel x13, x13, xzr, eq       // idx = dc ? 0 : zzi
    add x17, x7, x13, lsl #2     // Calculate address: X7 + (X13 << 2)
    ldr w14, [x17]               // W14 = unquant_mf[idx]
    sxtw x13, w14                // X13 (X_unquant_mf_val_64)
    dup z7.d, x13
    mov z2.d, z1.d
    mul z2.d, z2.d, z7.d

    cmp var_DC, #0               // Compare dc (W19) with 0
    b.ne q_not_zero_dc_not_zero
    srshr z2.d, P_TRUE/M, z2.d, #8
    b q_not_zero_dc_not_zero_done
q_not_zero_dc_not_zero:
    srshr z2.d, P_TRUE/M, z2.d, #7
q_not_zero_dc_not_zero_done:

    ldrsh x14, [x20, w15, sxtw #1]
    dup z3.d, x14
    abs z4.d, P_TRUE/M, z3.d
    sub z4.d, z4.d, z2.d
    mul z4.d, z4.d, z4.d

    cmp var_DC, #0                 // if (!dc (W19))
    b.ne dc_psy_check_failed_vl128
    cmp W21, #0                    // if (i_psy_trellis (W21))
    b.eq i_psy_trellis_check_failed_vl128
    cmp w0, #0                     // if (i (W0) != 0)
    b.eq i_is_zero_psy_check_failed_vl128

    ldrsh x13, [x22, w15, uxtw #1] // X13 = fenc_dct[zzi(W15)]
    sub x13, x13, x14              // X13 (X_pred_coef) = fenc_val - sign_coef(X14)
    mov z1.d, z2.d
    asr x17, x14, #63              // X17 (X_tmp_sign_bit) from sign_coef(X14)
    eor x14, x13, x17              // X14 = pred_coef(X13) ^ tmp_sign_bit
    sub x14, x14, x17              // X14 (X_signed_pred_coef) (SIGN64 result)
    dup z7.d, x14
    add z1.d, z1.d, z7.d
    abs z1.d, P_TRUE/M, z1.d

    ldr w14, [x23, w15, uxtw #2]   // coef_weight2[zzi(W15)]
    dup z7.d, x14
    mul z4.d, z4.d, z7.d

    ldr w17, [x24, w15, uxtw #2]   // coef_weight1[zzi(W15)]
    mul w13, w17, w21              // W13 (W_tmp_scalar_mul) = coef_w1 * i_psy_trellis(W21)
    dup z7.d, x13
    mls z4.d, P_TRUE/M, z1.d, z7.d
    B psy_block_done_vl128
dc_psy_check_failed_vl128:
i_psy_trellis_check_failed_vl128:
i_is_zero_psy_check_failed_vl128:
    cmp var_DC, #0                 // if (dc (W19))
    b.eq not_dc_block_vl128
    lsl z4.d, P_TRUE/M, z4.d, #8
    b psy_block_done_vl128
not_dc_block_vl128:
    // W15 still holds zzi
    ldr w14, [x23, w15, uxtw #2]   // X14 = coef_weight2[zzi(W1)]
    dup z7.d, x14
    mul z4.d, z4.d, z7.d
psy_block_done_vl128:

    cmp var_DC, #0                 // if (!dc (W19))
    b.ne i_is_zero_dc_check_failed_vl128
    cmp W0, #0                     // if (i (W0) == 0)
    b.ne i_is_zero_dc_check_failed_vl128
    mov z7.d, z4.d
    cmlt v10.2d, v3.2d, #0
    eor z0.d, z2.d, z10.d
    sub z0.d, z0.d, z10.d
    dup z8.d, #8
    add z0.d, Z0.d, Z8.d
    dup z8.d, #15
    bic z0.d, z0.d, z8.d
    mov z10.d, z3.d
    sub z10.d, z10.d, z0.d
    mul z10.d, z10.d, z10.d
    // W15 still holds zzi
    add x17, x23, w15, uxtw #2     // addr for coef_weight2[zzi(W15)]
    ldr w17, [x17]                 // Re-load W17 with coef_weight2 value
    dup z8.d, x17
    mul z10.d, z10.d, z8.d
    mov z4.d, z10.d
    B ssd0_done_vl128
i_is_zero_dc_check_failed_vl128:
    mov z7.d, z4.d
ssd0_done_vl128:

    ldr CUR_SCORE02, [CUR_SCORE02_PTR]
    mov SCORE02.d, CUR_SCORE02.d
    // P5 will hold prev_j02
    cmpge p5.d, P_TRUE/Z, CUR_SCORE02.d, #0  // Compare cur_score02 with 0
    orr   p5.b, P_TRUE/Z, p5.b, P_S0.b       // OR the result with s0
    mov TMP_P6.b, p5.b

    cmpge p4.d, P_TRUE/Z, CUR_SCORE13.d, #0
    mov TMP_P7.b, p4.b
    mov PREV_SCORE02.d, SCORE02.d
    mov PREV_SCORE13.d, CUR_SCORE13.d

    ldr x13, [x29, #184]
    ldr w14, [x13]                 // W14 (W_cur_cabac3_val)

    lsr w15, w14, #8               // W15 (W_lev1_state3)
    and w15, w15, #0xFF
    mov w8, w15
    sub w13, w15, #1               // W13 (W_trans_idx)

    lsl x13, x15, #1
    sub w13, w13, #1               // W13 (W_trans_idx)
    ldr w24, [x12, x13]            // W24 = W_transition3

    sub w13, w26, #1               // X13 (X_num_coefs_minus_1) from num_coefs(W26)
    cmp w0, w13, sxtw              // Compare i(W0)
    b.ge i_is_last_coef_vl128

    mov w15, w0                    // W15 (W_last_idx_calc) = i(W0)
    cmp var_DC, #0                 // dc(W19)
    b.ne dc_is_true_last_idx_vl128
    cmp w26, #64                   // num_coefs(W26)
    b.ne not_64_coeffs_last_idx_vl128
    ldrb w15, [x11, w0, uxtw]      // W15 (W_last_idx_calc) (X11 base, W0 is i)
    b last_idx_done_vl128
dc_is_true_last_idx_vl128:
    cmp w26, #8
    b.ne not_8_coeffs_dc_last_idx_vl128
    lsr w15, w0, #1
    mov w13, #2                    // W13 (W_two_li)
    cmp w15, w13
    csel w15, w15, w13, lt
    B last_idx_done_vl128
not_64_coeffs_last_idx_vl128:
not_8_coeffs_dc_last_idx_vl128:
last_idx_done_vl128:
    // State: W15=last_idx, W18=sig, X10=&entropy_table, [X29,#208]=&cabac_state_last
    ldr x13, [x29, #224]           // Load pointer to cabac_state_last.
    ldrb w13, [x13, w15, uxtw]     // W13 = cabac_state_last[last_idx].

    // cost_sig1 = x264_cabac_entropy[sig ^ 1], store to W15
    eor w14, w18, #1               // W14 = sig ^ 1
    ldrh w20, [x10, w14, uxtw #1]  // W20 = cost_sig1

    ldrh w14, [x10, w13, uxtw #1]                // W14 = entropy[last].
    add w14, w20, w14              // W14 = cost_sig1 (W20) + entropy[last] (W14)

    eor w17, w13, #1
    ldrh w17, [x10, w17, uxtw #1]                // entropy[last ^ 1].
    add w17, w20, w17

    dup z6.s, w14
    insr z6.s, w17
    b cost_siglast_2111_done_vl128
i_is_last_coef_vl128:
    mov TMP_Z0.d, #0
    mov z6.s, #0
cost_siglast_2111_done_vl128:

    dup z8.h, z1.h[0]
    dup z9.h, z1.h[4]

    ldr  z2, [CUR_LEVEL0123_PTR]
    trn1 z8.h, z2.h, z8.h
    trn1 z9.h, z2.h, z9.h

    st1  {V8.4S-V9.4S}, [X16]

    add x16, x16, #32
    str x16, [x4]

    cmp x28, #1                    // X28 is q
    b.ne q_not_one_vl128

    fmov x13, d4                   // X_ssd1_lane0
    dup z8.d, x13
    mov TMP_Z1.d, TMP_Z0.d
    add TMP_Z1.d, TMP_Z1.d, z8.d
    sub z7.d, z7.d, TMP_Z1.d
    sub z4.d, z4.d, TMP_Z1.d
    mov TMP_Z0.d, TMP_Z1.d

    mov CUR_SCORE02.d, SCORE02.d
    add CUR_SCORE02.d, P_S0/M, CUR_SCORE02.d, z7.d // cur_score += ssd0
    str CUR_SCORE02, [CUR_SCORE02_PTR]

    ldr z1, [LEV_USED0123_PTR]
    str z1, [CUR_LEVEL0123_PTR]

    add z1.s, z1.s, #4
    str z1, [LEV_USED0123_PTR]

    add x17, x10, w8, uxtw #1     // X17 = addr for x264_cabac_entropy[lev1_state3(W15)]
    ldrh w14, [x17]               // W14 = W_entropy3
    ldr x13, [x29, #192]          // X13 = ptr_entropy012_ (arg 19)
    ldr z3, [x13]
    mov v3.s[3], w14
    str z3, [x13]

    TRELLIS_COEF1_0_SVE_(1)

    B loop_continue_vl128

q_not_one_vl128:
    dup z0.d, #-1
    mov CUR_SCORE02.d, z0.d
    str CUR_SCORE02, [CUR_SCORE02_PTR] // ptr_cur_score02
    ldr x13, [x29, #168]          // X13 = ptr_cur_score13
    mov CUR_SCORE13.d, z0.d
    str z0, [x13]
    ldr x13, [x29, #176]          // X13 = ptr_cur_score46
    str z0, [x13]

    // W15 was lev1_state3
    eor w13, w8, #1               // W13 (W_lev1_state3_xor_1)
    ldrh w14, [x10, x13, lsl #1]  // W14 = W_entropy3xor
    ldr x13, [x29, #240]          // X13 = ptr_entropy012_xor (arg 25)
    ldr z2, [x13]
    mov v2.s[3], w14
    str z2, [x13]

    cmp x28, #2
    b.ne q_not_two_vl128

    ldrh w13, [x10, x8, lsl #1]   // W13 = W_entropy3_q2
    ldr x15, [x29, #192]          // X15 = ptr_entropy012_
    ldr z3, [x15]
    mov v3.s[3], w13
    str z3, [x15]

    TRELLIS_COEF1_0_SVE_(0)

    // W24 was W_transition3
    lsr w24, w24, #8
    mov w13, #1
    whilelo P_S0.s, wzr, w13      // P1 = 's0': {T,F,F,F}
    mov p5.b, TMP_P6.b
    mov p4.b, TMP_P7.b
    pfalse P_FALSE.b

    TRELLIS_COEF2_0_SVE_

    B phase2_switch_and_break_vl128

q_not_two_vl128:
    // W24 was W_transition3
    lsr w24, w24, #8
    sub w13, w28, #1              // X13 (X_q_minus_1)
    TRELLIS_COEFN_0_SVE_ 0, w13
    // X28 was X_q
    TRELLIS_COEFN_0_SVE_ 1, w28

phase2_switch_and_break_vl128:
    mov w13, #1                   // W13 (switch_to_phase2) = 1
    str w13, [sp, #96]
    B phase1_loop_end_vl128

loop_continue_vl128:
    sub w0, w0, #1                // i-- (W0 is i)
    B phase1_loop_condition

phase1_loop_end_vl128:
    ldr   w13, [sp, #96]
    ldr   x17, [x29, #288] // phase2_needed pointer
    str   w13, [x17]

    // Epilogue
    ldp x19, x20, [sp, #16]
    ldp x21, x22, [sp, #32]
    ldp x23, x24, [sp, #48]
    ldp x25, x26, [sp, #64]
    ldp x27, x28, [sp, #80]
    ldp x29, x30, [sp], #112 // Restore FP, LR and deallocate stack frame
    ret
endfunc

    #define SSD_EITHER Z9
    #define SCORE46 Z28
    #define SCORE57 Z27
.macro TRELLIS_COEF1_1_SVE_ stage:req
    // Temporary register aliases for this macro's scope
    #define TMP4567 Z16
    #define SCORE11 Z25
    #define PREV_J_2 P3
    #define PREV_J13 P6
    #define PREV_J46 P7
    #define PREV_J57 P8

    // ssd_either = svdup_lane_u64( ssd, stage );
    // SSD_VEC is Z14
    dup SSD_EITHER.d, Z14.d[\stage]

    // Calculate cost constant: cost_siglast_1 + (1 << CABAC_SIZE_BITS)
    // cost_siglast_1 is in W13 from the previous block
    mov w27, #(1 << 8)
    add w27, w27, w13

    // tmp0123 = ...
    // *entropy012_ is in Z3
    mov z28.s, w27
    ldr x16,   [x29, #240]
    ldr z3, [x16]
    add z12.s, z3.s, z28.s
    // tmp4567 = ...
    // Assuming Z1 holds entropy4567
    add TMP4567.s, z1.s, z28.s

    // score... = svadd_u64_x( p, prev_score..., ssd_either );
    // prev_score vectors are in Z20-Z23
    add     SCORE02.d, z20.d, SSD_EITHER.d
    add     SCORE13.d, z21.d, SSD_EITHER.d
    add     SCORE46.d, z22.d, SSD_EITHER.d
    add     SCORE57.d, z23.d, SSD_EITHER.d

    // score... = svsra_n_u64(...)
    ldr x27, [x29, #280]
    dup     z13.s, w27
    umullb  z19.d, z12.s, z13.s
    usra    SCORE02.d, z19.d, #4
    umullt  z19.d, z12.s, z13.s
    usra    SCORE13.d, Z19.d, #4
    umullb  z19.d, TMP4567.s, Z13.s
    usra    SCORE46.d, Z19.d, #4
    umullt  z19.d, TMP4567.s, Z13.s
    usra    SCORE57.d, Z19.d, #4

    // score11 = svdup_lane_u64( score13, 0 );
    dup     SCORE11.d, SCORE13.d[0]

    mov p3.b, p9.b
    // Predicate calculations
    #define P_PREV_J_1 P9
    #define P_PREV_J_3 P10
    pfalse p0.b
    uzp1   P_PREV_J_1.d, p0.d, PREV_J13.d
    ptrue  p0.b, vl16
    bic    P_PREV_J_3.b, p0/z, PREV_J13.b, P_S0.b

    // Score updates
    #define P_J_2 P11
    #define P_J_1 P12
    #define P_J_3 P13
    #define P_J46 P14
    #define P_J57 P15
    mov P_J_2.b, PREV_J_2.b
    ldr X17, [x29, #216]   // cur_score13 ptr
    ldr CUR_SCORE13, [x17]
    .if \stage
        mov   p0.b, P_J_2.b
        cmphi P_J_2.d, p0/z, CUR_SCORE13.d, SCORE02.d
    .endif
    umin    CUR_SCORE13.d, PREV_J_2/m, CUR_SCORE13.d, SCORE02.d

    mov P_J_1.b, P_PREV_J_1.b
    ldr x17, [x29, #208]
    ldr CUR_SCORE02, [x17]    // cur_score02 -> Z12
    .if \stage
        mov   p0.b, P_J_1.b
        cmphi P_J_1.d, p0/z, CUR_SCORE02.d, SCORE11.d
    .endif
    mov  p0.b, P_PREV_J_1.b
    umin CUR_SCORE02.d, p0/m, CUR_SCORE02.d, SCORE11.d
    str  CUR_SCORE02, [x17]   // cur_score13 -> Z1

    mov   P_J_3.b, P_PREV_J_3.b
    mov   p0.b, P_J_3.b
    cmphi P_J_3.d, p0/z, CUR_SCORE13.d, SCORE13.d
    mov   p0.b, P_PREV_J_3.b
    umin  CUR_SCORE13.d, p0/m, CUR_SCORE13.d, SCORE13.d
    ldr   x17, [x29, #216]
    str   CUR_SCORE13, [x17]  // cur_score13 -> Z1

    mov  P_J46.b, PREV_J46.b
    ldr  x17, [X29, #224]
    ldr  z30, [x17]   // cur_score46 -> Z30
    .if \stage
        mov   p0.b, P_J46.b
        cmphi P_J46.d, p0/z, z30.d, SCORE46.d
    .endif
    mov  p0.b, PREV_J46.b
    umin z30.d, p0/m, z30.d, SCORE46.d
    str  z30, [x17]   // cur_score46

    mov  P_J57.b, PREV_J57.b
    .if \stage
        mov   p0.b, P_J57.b
        cmphi P_J57.d, p0/z, Z31.d, SCORE57.d
    .endif
    mov  p0.b, PREV_J57.b
    umin z31.d, p0/m, z31.d, SCORE57.d

    pfalse p0.b

    // Final predicate combinations
    #define P_J_2NOT3 P9
    #define P_J__12   P10
    #define P_J___3   P11
    #define P_J4567   P12
    bic     P_J_2NOT3.b, p2/z, P_J_2.b, P_J_3.b
    trn1    P_J__12.s, P_J_1.s, P_J_2.s
    uzp1    P_J___3.s, p0.s, P_J_3.s
    trn1    P_J4567.s, P_J46.s, P_J57.s

    // CABAC state updates
    ptrue P_TRUE.b, vl16
    ptest P_TRUE, P_J_2NOT3.b
    ldr   x17, [x29, #232]  // x17 = cur_cabac3
    ldr   w26, [x17]
    ldr   w25, [x29, #296]  // level_state0489
    csel  w26, w25, w26, ne // cur_cabac3 = svptest_any(p, j_2not3) ? level_state0489 : cur_cabac3

    eor     w25, w26, w24 // cur_cabac3 ^ transition3
    and     w25, w25, #0xff00
    eor     w25, w26, w25
    ptest   P_TRUE, P_J_3.b
    csel    w26, w25, w26, ne
    str     w26, [x17]

    sel     z6.b, P_J4567, z8.b, z6.b

    // Level index updates
    ldr     z27, [x3] // cur_level0123
    ldr x9, [x29, #312] // arg 27: cur_level4567_ptr
    ldr     z12, [x9]
    ldr     z13, [x5] // lev_used0123

    mov     z30.d, z13.d
    sub     z30.s, z30.s, #1
    sel     z27.s, P_J__12, z30.s, z27.s

    mov     z30.d, z13.d
    add     z30.s, z30.s, #4
    sel     z12.s, P_J4567, z30.s, z12.s

    sel     z27.s, P_J___3, z13.s, z27.s

    add     z13.s, z13.s, #8
    str     z27, [x3] // cur_level0123
    str     z12, [x9] // cur_level4567
    str     z13, [x5] // lev_used0123
    mov p9.b, p3.b
    pfalse P_FALSE.b

.endm

.macro TRELLIS_COEF2_1_SVE_
    // const svuint64_t ssd_either = svdup_lane_u64( ssd, 1 );
    dup         SSD_EITHER.d, z14.d[1]

    // const svuint32_t size_unary0123 = svdup_u32( cabac_size_unary_5[128] );
    ldr  x17, [x29, #320] // x264_cabac_size_unary_ptr
    mov  x27, #256        // 128 * sizeof (uint16_t)
    ldrh w9, [x17, x27]
    dup  z12.s, w9

    // const svuint32_t size_unary4567 = svld1uh_gather_u32index_u32(p, x264_cabac_size_unary[1], lgt1_state4567);
    movrel x17, X(cabac_size_unary) // x17 = &x264_cabac_size_unary
    add  x17, x17, #256
    ld1h {z3.s}, P_TRUE/z, [x17, z2.s, uxtw #1]

    // Calculate cost components (f8_bits)
    ldr  x17, [x29, #288]
    mov  z5.d, z8.d
    ldr  z8, [x17]
    dup  z28.s, w13
    add  z8.s, z8.s, z28.s
    add  z8.s, z8.s, z12.s // z8 = tmp0123

    add  z7.s, z4.s, z28.s
    add  z7.s, z7.s, z3.s  // z7 = tmp4567

    // score = nodes_prev->score[j] + ssd;
    // prev_score vectors are in Z20-Z23
    add  SCORE02.d, z20.d, SSD_EITHER.d
    add  SCORE13.d, z21.d, SSD_EITHER.d
    add  SCORE46.d, z22.d, SSD_EITHER.d
    add  SCORE57.d, z23.d, SSD_EITHER.d

    // score += f8_bits * lambda2 >> ( CABAC_SIZE_BITS - LAMBDA_BITS );
    ldr x27, [x29, #280]
    dup      z3.s, w27                   // Broadcast lambda2
    umullb   z12.d, z8.s, z3.s
    usra     SCORE02.d, z12.d, #(CABAC_SIZE_BITS - LAMBDA_BITS)
    umullt   z12.d, z8.s, z3.s
    usra     SCORE13.d, z12.d, #(CABAC_SIZE_BITS - LAMBDA_BITS)
    umullb   z12.d, z7.s, z3.s
    usra     SCORE46.d, z12.d, #(CABAC_SIZE_BITS - LAMBDA_BITS)
    umullt   z12.d, z7.s, z3.s
    usra     SCORE57.d, z12.d, #(CABAC_SIZE_BITS - LAMBDA_BITS)

    // Prepare scores for comparison
    dup      z12.d, SCORE02.d[1]               // score22
    dup      z3.d, SCORE13.d[1]               // score33
    zip1     z8.d, z3.d, z27.d          // score35

    // save p6-9 to z16-19
    mov z0.b, #0
    mov z1.b, #-1
    sel z16.b, p6, z1.b, z0.b
    sel z17.b, p7, z1.b, z0.b
    sel z18.b, p8, z1.b, z0.b
    sel z19.b, p9, z1.b, z0.b

    // Prepare predicates for score updates.
    uzp2   p9.d, p9.d, P_FALSE.d          // p9 = prev_j2
    bic    p10.b, P_TRUE/z, p8.b, P_S0.b  // p10 = prev_j_7
    uzp2   p11.d, p6.d, p6.d
    uzp1   p11.d, p11.d, p8.d             // p11 = prev_j35
    and    p8.b, P_TRUE/z, P_S0.b, p6.b   // p8 = prev_j1

    // Compare and update scores, modifying predicates for winning lanes
    ldr    z26, [x10]           // z26 = *cur_score46
    mov    p12.b, p8.b
    mov    p0.b, p8.b // clobbered P_TRUE, using p0 as temp
    cmphi  p12.d, p0/z, z26.d, SCORE13.d
    umin   z26.d, p0/m, z26.d, SCORE13.d  // j1

    mov    p13.b, p7.b
    cmphi  p13.d, p7/z, z31.d, SCORE46.d
    umin   z31.d, p7/m, z31.d, SCORE46.d  // j46

    mov    p0.b, p9.b
    cmphi  p6.d, p0/z, z26.d, z12.d
    umin   z26.d, p0/m, z26.d, z12.d      // j2

    mov    p0.b, p10.b
    cmphi  p15.d, p0/z, z31.d, SCORE57.d
    umin   z31.d, p0/m, z31.d, SCORE57.d  // j_7

    mov    p0.b, p11.b
    cmphi  p8.d, p0/z, z26.d, z8.d
    umin   z26.d, p0/m, z26.d, z8.d      // j35
    ptrue  P_TRUE.b, vl16 // restored clobbered P_TRUE

    str    z26, [x10]

    // Combine winning predicates for subsequent operations
    orr    p9.b, P_S0/z, p12.b, p6.b       // p9 = j1 | j2
    orr    p9.b, P_S0/z, p9.b, p8.b        // p9 = j1or2or3
    and    p10.b, P_TRUE/z, P_S0.b, p8.b   // p10 = j3
    trn2   p11.d, p8.d, p15.d              // p11 = j57
    trn1   p8.s, p8.s, p13.s               // p8 = j3456, can reuse p8
    trn1   p13.s, p13.s, p11.s             // p13 = j4567
    and    p14.b, P_TRUE/z, P_S23.b, p13.b // p14 = j__67
    and    p15.b, P_TRUE/z, P_S3.b, p13.b  // p15 = j___7
    trn1   p7.b, P_FALSE.b, p10.b          // p7 = j_3______________

    // Update cur_cabac4567 based on winning paths
    ldr w15, [x29, #296]        // w15 = level_state0489
    dup z24.s, w15
    sel z6.s, p9, z24.s, z6.s
    mov v24.b[0], w24           // transition3 is in w24
    mov z6.b, p7/m, b24        // cur_cabac4567

    // new_cabac4567 = svsel(j4567, transition4567, prev_cabac4567)
    sel z12.b, p13, z5.b, z11.b        // z12 = new_cabac4567

    // Handle transition_unary updates
    // Assumes x12 holds base of &x264_cabac_transition_unary[1][-2]
    movrel x17, X(cabac_transition_unary)
    add  x17, x17, #126 // 126 = index [1][-2]
    mov  p7.b, p14.b

    // Conditional update based on 'dc' flag
    cmp         var_DC, #0
    b.eq        .L_else_dc
    trn1        p0.h, P_FALSE.h, p14.h
    ld1w z3.s,  p7/z, [x17, z2.s, uxtw] // z3 = transition_unary__67
    sel         z12.b, p0, z3.b, z12.b
    ptrue p0.b, vl16 // restore P0

    b           .L_endif_dc
.L_else_dc:
    add z2.s, P_S3/m, z2.s, z1.s
    ld1w z3.s, p7/z, [x17, z2.s, uxtw] // z3 = transition_unary__67
    mov z24.s, #0xff0000
    dup z8.s,  #0
    mov z8.s, p14/m, z24.s
    lsl z8.s, P_S3/m, z8.s, #8
    // svbsl_u32( transition_unary__67, new_cabac4567, tmp4567 );
    bsl z3.d,z3.d,z12.d,z8.d
    mov z12.d, z3.d
.L_endif_dc:

    // Finalize cur_cabac4567 updates
    mov    z8.d, z6.d
    ext    z8.b, z8.b, z8.b, #4
    sel    z8.s, p13, z12.s, z8.s
    index  z7.s, #-1, #1
    tbx    z6.s, z8.s, z7.s
    sel    z6.s, p15, z12.s, z6.s

    // Update level indices based on winning paths
    ldr  x17, [x29, #312]
    ldr  z24, [x17] // ptr_cur_level4567, z24 will hold the result
    ldr  z30, [x5]  // lev_used0123, z30 will hold the result
    dup  z3.s, z30.s[1]
    sel  z24.s, p12, z3.s, z24.s  // Update for j1

    dup  z3.s, z30.s[2]
    sel  z24.s, p6, z3.s, z24.s   // Update for j2

    mov  z3.d, z30.d
    add  z3.s, z3.s, #3
    sel  z24.s, p8, z3.s, z24.s

    mov  z3.d, z30.d
    add  z3.s, z3.s, #4
    sel  z24.s, p15, z3.s, z24.s
    str  z24, [x17]

    // Update total levels used count + 8
    add  z30.s, z30.s, #8
    str  z30, [LEV_USED0123_PTR]

    // restore p6-9 from z16-19
    cmpne p6.b, P_TRUE/z, z16.b, #0
    cmpne p7.b, P_TRUE/z, z17.b, #0
    cmpne p8.b, P_TRUE/z, z18.b, #0
    cmpne p9.b, P_TRUE/z, z19.b, #0
.endm

.macro TRELLIS_COEFN_1_SVE_ stage:req, abs_level:req
    // const svuint64_t ssd_either = svdup_lane_u64( ssd, stage );
    dup  SSD_EITHER.d, z14.d[\stage]

    // const int prefix = X264_MIN( (abs_level)-1, 14 );
    sub    w26, \abs_level, #1
    mov    w18, #14
    cmp    w26, w18
    csel   w26, w26, w18, le                 // w26 = prefix

    // const int suffix_cost = (abs_level) >= 15 ? ...
    mov    w18, wzr                        // w4 = suffix_cost = 0
    cmp    \abs_level, #15
    blt    .L_SKIP_SUFFIX_CALC_\@
    bs_size_ue_big_macro \abs_level      // Assume this macro calculates size in w18
    lsl    w18, w18, #CABAC_SIZE_BITS
.L_SKIP_SUFFIX_CALC_\@:

    // const svuint32_t size_unary0123 = svdup_u32( cabac_size_unary_5[prefix * 128] );
    lsl  x27, x26, #8
    ldr  x17, [x29, #320]             // x17 = x264_cabac_size_unary_ptr
    ldrh w9, [x17, x27]
    dup  z12.s, w9                     // z12 = size_unary0123

    movrel x17, X(cabac_size_unary)
    // const svuint32_t size_unary4567 = svld1uh_gather_...
    add  x17, x17, x27
    ld1h {z3.s}, P_TRUE/z, [x17, z2.s, uxtw #1]   // Perform a standard scalar load from base + (index * 2)

    // Cost calculation
    add  w9, w13, w18       // w9 = cost_siglast_1 + suffix_cost
    ldr  x17, [x29, #288]   // x17 = entropy012_xor_ptr
    mov  z5.d, z8.d
    ldr  z8, [x17]
    dup  z28.s, w9
    add  z8.s, z8.s, z28.s
    add  z8.s, z8.s, z12.s  // z6 = tmp0123

    add  z7.s, z4.s, z28.s
    add  z7.s, z7.s, z3.s   // z7 = tmp4567

    // Score calculation: score = prev_score + ssd
    add SCORE02.d, z20.d, SSD_EITHER.d // score02
    add SCORE13.d, z21.d, SSD_EITHER.d // score13
    add SCORE46.d, z22.d, SSD_EITHER.d // score46
    add SCORE57.d, z23.d, SSD_EITHER.d // score57

    // score += f8_bits * lambda2 >> shift
    ldr x27, [x29, #280]
    dup    z3.s, w27                      // z3 = lambda2
    umullb z12.d, z8.s, z3.s
    usra   SCORE02.d, z12.d, #(CABAC_SIZE_BITS - LAMBDA_BITS)
    umullt z12.d, z8.s, z3.s
    usra   SCORE13.d, z12.d, #(CABAC_SIZE_BITS - LAMBDA_BITS)
    umullb z12.d, z7.s, z3.s
    usra   SCORE46.d, z12.d, #(CABAC_SIZE_BITS - LAMBDA_BITS)
    umullt z12.d, z7.s, z3.s
    usra   SCORE57.d, z12.d, #(CABAC_SIZE_BITS - LAMBDA_BITS)

    // Vector setup for comparisons
    dup    z12.d, SCORE02.d[1]  // z12 = score22
    dup    z3.d,  SCORE13.d[1]  // z3 = score33
    zip1   z8.d, z3.d, z27.d    // z8 = score35

    // Predicate generation

    .if \stage
    // if stage 0
    // restore p6-9 from z16-19
    cmpne p6.b, P_TRUE/z, z16.b, #0
    cmpne p7.b, P_TRUE/z, z17.b, #0
    cmpne p8.b, P_TRUE/z, z18.b, #0
    cmpne p9.b, P_TRUE/z, z19.b, #0
    .else
    // save p6-9 to z16-19
    mov z0.b, #0
    mov z1.b, #-1
    sel z16.b, p6, z1.b, z0.b
    sel z17.b, p7, z1.b, z0.b
    sel z18.b, p8, z1.b, z0.b
    sel z19.b, p9, z1.b, z0.b
    .endif

    uzp2   p9.d, p9.d, P_FALSE.d               // p9 = prev_j2
    bic    p10.b, P_TRUE/z, p8.b, P_S0.b        // p10 = prev_j_7
    uzp2   p11.d, p6.d, p6.d
    uzp1   p11.d, p11.d, p8.d             // p11 = prev_j35
    // save p8 to z18
    and    p8.b, P_TRUE/z, P_S0.b, p6.b         // p8 = prev_j1

    // Score comparisons and updates
    ldr    z26, [x10]           // z26 = *cur_score46

    mov    p12.b, p8.b                    // p12 = j1
    mov p0.b, p8.b // clobbered P_TRUE, using p0 as temp
    .if \stage
      cmphi  p12.d, p0/z, z26.d, SCORE13.d    // j1 = score13 < *cur_score46
    .endif
    umin   z26.d, p0/m, z26.d, SCORE13.d

    mov    p13.b, p7.b                    // p13 = j46
    .if \stage
      cmphi  p13.d, p7/z, z31.d, SCORE46.d    // j46 = score46 < cur_score57
    .endif
    umin   z31.d, p7/m, z31.d, SCORE46.d

    mov p0.b, p9.b
    cmphi  p6.d, p0/z, z26.d, z12.d       // j2 = score22 < *cur_score46
    umin   z26.d, p0/m, z26.d, z12.d

    mov    p0.b, p10.b                   // p0 = j_7
    cmphi  p15.d, p0/z, z31.d, SCORE57.d     // j_7 = score57 < cur_score57
    umin   z31.d, p0/m, z31.d, SCORE57.d

    mov    p0.b, p11.b                    // p8 = j35
    cmphi  p8.d, p0/z, z26.d, z8.d       // j35 = score35 < *cur_score46
    umin   z26.d, p0/m, z26.d, z8.d
    ptrue  P_TRUE.b, vl16 // restored clobbered P_TRUE

    str    z26, [x10]

    // Further predicate manipulation
    orr    p9.b, P_S0/z, p12.b, p6.b         // p9 = j1 | j2
    orr    p9.b, P_S0/z, p9.b, p8.b          // p9 = j1or2or3
    and    p10.b, P_TRUE/z, P_S0.b, p8.b     // p10 = j3
    trn2   p11.d, p8.d, p15.d                // p11 = j57
    trn1   p8.s, p8.s, p13.s                 // p8 = j3456, can reuse p8
    trn1   p13.s, p13.s, p11.s               // p13 = j4567
    and    p14.b, P_TRUE/z, P_S23.b, p13.b   // p14 = j__67
    and    p15.b, P_TRUE/z, P_S3.b, p13.b    // p15 = j___7
    trn1   p7.b, P_FALSE.b, p10.b            // p7 = j_3______________

    // CABAC state update
    mov  z30.d, CUR_CABAC4567.d
    ldr  w15, [x29, #296]        // w15 = level_state0489
    dup  z24.s, w15
    sel  z30.s, p9, z24.s, z30.s
    mov  v24.b[0], w24           // transition3 is in w24
    mov  z30.b, p7/m, b24        // cur_cabac4567
    mov z6.d, z30.d

// new_cabac4567 = svsel(j4567, transition4567, prev_cabac4567)
    sel  z12.b, p13, z5.b, z11.b // z12 = new_cabac4567

    movrel x17, X(cabac_transition_unary)
    lsl  x8, x26, #7    // base + prefix * 128
    sub  x8, x8, #2     // base[prefix][-2]
    add  x17, x17, x8
    mov  p7.b, p14.b

    cmp    var_DC, #0
    beq    .L_ELSE_DC_\@
    // if(dc)
    trn1   p0.h, P_FALSE.h, p14.h      // p12 = j_____6_7
    ld1w z3.s, p7/z, [x17, z2.s, uxtw] // z3 = transition_unary__67
    sel    z12.b, p0, z3.b, z12.b
    ptrue p0.b, vl16 // restore P0 = ptrue again
    b      .L_END_DC_IF_\@
.L_ELSE_DC_\@:
    add z2.s, P_S3/m, z2.s, z1.s
    ld1w z3.s, p7/z, [x17, z2.s, uxtw] // z3 = transition_unary__67
    sub z2.s, P_S3/m, z2.s, z1.s
    mov z24.s, #0xff0000
    dup z8.s,  #0
    mov z8.s, p14/m, z24.s
    mov z24.s, #8
    lsl z8.s, P_S3/m, z8.s, z24.s
 // svbsl_u32( transition_unary__67, new_cabac4567, tmp4567 );
 // only bsl u64 is available, hence
    bsl z3.d,z3.d,z12.d,z8.d
    mov z12.d, z3.d
.L_END_DC_IF_\@:

    mov    z8.d, z30.d
    ext    z8.b, z8.b, z8.b, #4         // z6 = svext(...)
    sel    z8.s, p13, z12.s, z8.s          // z6 = tmp4567
    index  z7.s, #-1, #1
    tbx    z30.s, z8.s, z7.s
    sel    z30.s, p15, z12.s, z30.s
    mov    CUR_CABAC4567.d, z30.d

    ldr x17, [x29, #312]
    ldr z24, [x17]  // ptr_cur_level4567, z24 will hold the result
    ldr z30, [x5]  // lev_used0123, z30 will hold the result
    dup z3.s, z30.s[1]
    sel z24.s, p12, z3.s, z24.s

// for j2 (predicate p6)
    dup  z3.s, z30.s[2]
    sel  z24.s, p6, z3.s, z24.s

    mov  z3.d, z30.d
    add  z3.s, z3.s, #3
    sel  z24.s, p8, z3.s, z24.s
    mov  z3.d, z30.d
    add  z3.s, z3.s, #4
    sel  z24.s, p15, z3.s, z24.s

    str  z24, [x17]

    // *lev_used0123 += 8;
    add  z30.s, z30.s, #8
    str  z30, [LEV_USED0123_PTR]
.endm

#define CUR_SCORE13_PTR   X9
#define CUR_SCORE46_PTR   X10
#define CUR_LEVEL4567_PTR X11
#define var_I_PSY_TRELLIS W21
#define LEVEL_STATE6767   Z10 // duplicated level state 6
#define CUR_SCORE57       Z31 // initial max score for levels 5,7
function trellis_cabac_phase2, export=1
    stp x29, x30, [sp, #-160]!
    mov x29, sp
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]
    stp x23, x24, [sp, #48]
    stp x25, x26, [sp, #64]
    stp x27, x28, [sp, #80]
    sub sp, sp, #64          // Space for vector registers
    stp d8, d9,   [sp, #0]
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]

    // Load stack arguments into callee-saved registers
    // Args in GPRs: W0=i, X1=b_ac, X2=quant_coefs, X3=PTR_CUR_LEVEL0123,
    // X4=next_level, X5=lev_used0123, X6=zigzag, X7=unquant_mf
    ldp x25, x13, [x29, #208] // cur_score02_ptr, cur_score13_ptr
    ldr w12,      [x29, #336] // switched
    ldr w26,      [x29, #248] // num_coefs
    ldr var_DC,   [x29, #160] //
    ptrue P_TRUE.b, vl16
    pfalse P_FALSE.b
    cbz w12, .phase2_else_clause

    ldr x12,      [x29, #168] // orig_coefs
    ldr var_I_PSY_TRELLIS, [x29, #176] // arg 11
    ldp x22, x23, [x29, #184] // fenc_dct, coef_weight2
    ldr x24,      [x29, #200] // coef_weight1
    ldp x14, x15, [x29, #224] // cur_score46_ptr, cur_cabac3_ptr
    ldr x16,      [x29, #240] // entropy012_ptr
    // b_interlaced           // 256
    ldr x17,      [x29, #264] // arg 21: cabac_state_sig_ptr
    // cabac_state_last_ptr   // 272
    ldr varX_LAMBDA2, [x29, #280]
    // entropy012_xor         // 288
    // level_state0489        // 296
    ldp x10, x9,  [x29, #304] // cur_cabac4_ptr, cur_level4567_ptr
    // cabac_size_unary_5     // 320
    // level_state01234567    // 328
    // next                   // 344
    // dct                    // 352

    // Load global table addresses
    movrel x20, X264(cabac_entropy)
    movrel x21, X264(cabac_transition)

    // const svuint32_t level_state6767 = ...
    ldr x8, [x29, #328]        // level_state01234567 (arg 30)
    lsr x8, x8, #48            // Shift right to get state7|state6 in lower 16 bits
    mov LEVEL_STATE6767.h, w8  // Duplicate state6 into all 32-bit lanes
    uunpklo LEVEL_STATE6767.h, LEVEL_STATE6767.b
    uunpklo LEVEL_STATE6767.s, LEVEL_STATE6767.h

    // svuint64_t cur_score57 = ... TRELLIS_SCORE_MAX;
    // TRELLIS_SCORE_MAX is (uint64_t)-1
    dup CUR_SCORE57.d, #-1      // Initialize cur_score57 vector with max score

    // svuint32_t cur_cabac4567 = svdup_u32(cur_cabac4);
    ldr x8, [x29, #304]
    ldr w9, [x8]               // cur_cabac4
    dup CUR_CABAC4567.s, w9

    // SVE Predicate and initial state setup
    mov w8, #1
    whilelo P_S0.s, wzr, w8    // P_S0 = {T,F,F,F}
    index z12.s, #0, #1
    mov w8, #2
    dup z30.s, w8
    cmpeq P_S2.s, P_TRUE/z, z12.s, z30.s // P_S2 = {F,F,T,F}

    // const svbool_t s3 = svdupq_b32(false, false, false, true); // P_S3 = {F,F,F,T,...}
    mov w8, #3
    dup z8.s, w8
    cmpeq P_S3.s, P_TRUE/z, z12.s, z8.s
    // const svbool_t s23 = svdupq_b32(false, false, true, true);
    orr P_S23.b, P_TRUE/z, P_S2.b, P_S3.b

phase2_loop_start:
    // while(__builtin_expect( --i >= b_ac, 1 ))
    sub w0, w0, #1
    cmp w0, w1
    b.lt phase2_loop_end
    // SVE Predicate and initial state setup
    ptrue P_TRUE.b, vl16
    pfalse P_FALSE.b
    mov w8, #1
    whilelo P_S0.s, wzr, w8       // P_S0 = {T,F,F,F}
    index z12.s, #0, #1
    mov w8, #2
    dup z30.s, w8
    cmpeq P_S2.s, P_TRUE/z, z12.s, z30.s // P_S2 = {F,F,T,F}

    // const svbool_t s3 = svdupq_b32(false, false, false, true); // P_S3 = {F,F,F,T,...}
    mov w8, #3
    dup z8.s, w8
    cmpeq P_S3.s, P_TRUE/z, z12.s, z8.s
    // const svbool_t s23 = svdupq_b32(false, false, true, true);
    orr P_S23.b, P_TRUE/z, P_S2.b, P_S3.b

    // int64_t q = quant_coefs[i];
    ldrsh x11, [x2, w0, sxtw #1] // X11 = q
    cmp x11, #0
    b.ne q_not_zero_p2

// if(__builtin_expect( !q, 0 ))
q_is_zero_p2:
    ldr x10, [x4]
    ldr z0, [CUR_LEVEL0123_PTR]
    ldr x9, [x29, #312] // cur_level4567_ptr
    ldr z1, [x9]
    st1 {v0.4s, v1.4s}, [x10]
    add x10, x10, #32 // 2 * 16 bytes
    str x10, [x4]

    ldr z0, [CUR_LEVEL0123_PTR]
    ldr z2, [LEV_USED0123_PTR]
    sel z0.s, P_S0, z0.s, z2.s
    str z0, [CUR_LEVEL0123_PTR]

    mov z1.d, z2.d
    add z1.s, z1.s, #4
    str z1, [x9]

    add z2.s, z2.s, #8
    str z2, [LEV_USED0123_PTR]
    b phase2_loop_start

q_not_zero_p2:
    cneg x11, x11, lt    // q = labs(q)
    dup  z12.d, x11      // Z12 = abs(q) broadcasted
    mov  TMP_Z5.d, #1
    sub  z12.d, P_S0/m, z12.d, TMP_Z5.d  // abs_q = svsub_n_s64_m(s0, ...)

    // SSD Calculation Block - similar to phase1 but uses svsubr
    ldrb w12, [x6, w0, uxtw] // W12 = zzi = zigzag[i]
    mov  x13, xzr
    cmp  var_DC, #0
    csel x13, xzr, x12, ne   // idx = dc ? 0 : zzi

    ldrsw x14, [x7, x13, lsl #2]  // W14 = unquant_mf[idx]
    dup z13.d, x14
    mul z13.d, z13.d, z12.d // unquant_abs_lev

    cmp var_DC, #0
    b.ne dc_not_zero_p2_ssd
    srshr z13.d, P_TRUE/m, z13.d, #8
    b dc_shift_done_p2_ssd
dc_not_zero_p2_ssd:
    srshr z13.d, P_TRUE/m, z13.d, #7
dc_shift_done_p2_ssd:

    ldr x8, [x29, #168]           // orig_coefs
    ldrsh x14, [x8, w12, sxtw #1] // sign_coef
    cmp x14, #0
    cneg x15, x14, lt

    dup z14.d, x15
    sub z14.d, z14.d, z13.d // dd = svsubr_n_s64_x(...)
    mul z14.d, z14.d, z14.d // dd = svmul_s64_x(...)

    // Psy trellis block
    cmp var_DC, #0
    b.ne psy_check_failed_p2
    ldr var_I_PSY_TRELLIS, [x29, #176]
    cmp var_I_PSY_TRELLIS, #0
    b.eq psy_check_failed_p2
    cmp W0, #0
    b.eq psy_check_failed_p2

    ldrsh x30, [x22, w12, sxtw #1] // fenc_dct[zzi]
    sub x30, x30, x11              // pred_coef
    mov z29.d, z13.d  // psy_value
    asr x28, x11, #63
    eor x26, x30, x28
    sub x26, x26, x28             // SIGN64
    dup z30.d, x26
    add z29.d, z29.d, z30.d
    abs z29.d, P_TRUE/m, z29.d

    ldrsw  x26, [x23, w12, uxtw #2]
    dup  z30.d, x26
    mul  z14.d,  z14.d, z30.d
    ldr  w28, [x24, w12, uxtw #2]
    mul  w26, w28, var_I_PSY_TRELLIS
    dup  z30.d, x26
    mls  z14.d, P_TRUE/m, z29.d, z30.d

    b psy_block_done_p2

psy_check_failed_p2:
    cmp var_DC, #0
    b.eq not_dc_block_p2
    lsl z14.d, P_TRUE/m, z14.d, #8
    b psy_block_done_p2
not_dc_block_p2:
    ldr  x23, [x29, #192]         // arg 12: coef_weight2
    ldr  w15, [x23, w12, uxtw #2] // coef_weight2[zzi]
    dup  z15.d, x15
    mul  z14.d, z14.d, z15.d

psy_block_done_p2:
    // Z14 contains the final ssd result.
    // Load current scores and check validity
    ldr CUR_SCORE02_PTR, [x29, #208] // Load cur_score02 ptr
    ldr CUR_SCORE13_PTR, [x29, #216] // Load cur_score13 ptr
    ldr CUR_SCORE46_PTR, [x29, #224] // Load cur_score46 ptr
    ldr CUR_SCORE02, [CUR_SCORE02_PTR]   // cur_score02 -> Z0
    ldr z1, [CUR_SCORE13_PTR]   // cur_score13 -> Z1
    ldr z2, [CUR_SCORE46_PTR]   // cur_score46 -> Z2
    // Z11 already has cur_score57

    cmpge p9.d, P2/Z, CUR_SCORE02.d, #0 // prev_j_2
    cmpge p6.d, P_TRUE/Z, z1.d, #0      // prev_j13
    cmpge p7.d, P_TRUE/Z, z2.d, #0      // prev_j46
    cmpge p8.d, P_TRUE/Z, z31.d, #0     // prev_j57

    // Store previous scores for use in TRELLIS macros
    mov z20.d, CUR_SCORE02.d
    mov z21.d, z1.d  // prev_score13
    mov z22.d, z2.d  // prev_score46
    mov z23.d, z31.d // prev_score57

    // Calculate CABAC states
    ldr x15, [x29, #232]  // arg 17: cur_cabac3_ptr
    ldrb w18, [x15, #1]   // *cur_cabac3
    mov z7.d, z6.d
    and z7.s, z7.s, #0xFF // lev1_state4567

    // Update entropy for lane 3
    ldrh w26, [x20, x18, lsl #1]   // x264_cabac_entropy[lev1_state3]
    ldr  x15, [x29, #272]          // arg 22: cabac_state_last_ptr
    ldr  z3, [x16]
    mov  v3.s[3], w26
    str  z3, [x16]

    // Get transition values
    lsl  x24, x18, #1
    sub  x24, x24, #1
    movrel x21, X264(cabac_transition)
    ldr  w24, [x21, x24]                        // transition3
    ld1h {z8.s}, P_TRUE/z, [x21, z7.s, uxtw #1] // transition4567

    mov  w14, w0
    // condition: !dc && num_coefs == 64
    cmp  var_DC, #0       // if (dc != 0) goto check for the second condition
    b.ne .dc_is_true_sig_idx_p2
    ldr  w26, [x29, #248] // arg 19: num_coefs
    cmp  w26, #64         // if (num_coefs != 64) goto the final else case
    b.ne .final_else_case_sig_idx_p2

    // Case 1: !dc && num_coefs == 64
    // sig_idx = x264_significant_coeff_flag_offset_8x8[b_interlaced][i];
    movrel x27, X264(significant_coeff_flag_offset_8x8)
    mov  w15, #64             // 64 coefficients per row in the table
    ldr  w17, [x29, #256]
    mul  w17, w17, w15        // row_offset = b_interlaced (W11) * 64
    add  x17, x27, w17, uxtw  // base_of_row = table_base (X18) + row_offset
    ldrb w14, [x17, w0, uxtw] // sig_idx = table[b_interlaced][i]
    b    .sig_idx_done_p2

.dc_is_true_sig_idx_p2:
    // condition: dc && num_coefs == 8
    cmp  w26, #8  // if (num_coefs != 8) goto the final else case
    b.ne .final_else_case_sig_idx_p2

    // Case 2: dc && num_coefs == 8
    // sig_idx = X264_MIN( i/2, 2 );
    lsr  w14, w0, #1          // W14 = i / 2
    mov  w15, #2              // The value to compare against
    cmp  w14, w15
    csel w14, w14, w15, lt    // sig_idx = min(i/2, 2)
    b    .sig_idx_done_p2

.final_else_case_sig_idx_p2:
    // This label is reached if neither of the special cases match.
    // The default value 'sig_idx = i' set at the beginning is used.

.sig_idx_done_p2:
    mov  w13, w0 // W13 will hold last_idx, default to i
    // Check for condition: !dc && num_coefs == 64
    cmp  var_DC, #0
    b.ne .dc_is_true_last_idx_p2
    cmp  w26, #64
    b.ne .final_else_case_last_idx_p2

    // Case 1: !dc && num_coefs == 64
    movrel x28, X264(last_coeff_flag_offset_8x8)
    ldrb w13, [x28, w0, uxtw]     // last_idx = table[i]
    b    .last_idx_done_p2

.dc_is_true_last_idx_p2:
    // Check for condition: dc && num_coefs == 8
    cmp  w26, #8
    b.ne .final_else_case_last_idx_p2

    // Case 2: dc && num_coefs == 8
    lsr  w13, w0, #1
    mov  w15, #2
    cmp  w13, w15
    csel w13, w13, w15, lt      // last_idx = min(i/2, 2)
    b    .last_idx_done_p2

.final_else_case_last_idx_p2:
    // Default case: last_idx = i (already in W13)

.last_idx_done_p2:
    // W13 now holds last_idx. W14 still holds sig_idx.
    ldr x15,  [x29, #264]
    ldrb w25, [x15, w14, uxtw]  // sig = cabac_state_sig[sig_idx]
    ldr x15,  [x29, #272]
    ldrb w26, [x15, w13, uxtw]  // last = cabac_state_last[last_idx]

    eor  w13, w25, #1
    ldrh w13, [x20, x13, lsl #1]
    ldrh w14, [x20, x26, lsl #1]
    add  w13, w13, w14          // cost_siglast_1

    // Update Level Tree
    dup z3.h, z12.h[0]
    dup z4.h, z12.h[4]
    ldr z0, [x3]
    ldr x9, [x29, #312] // arg 27: cur_level4567_ptr
    ldr  z1, [x9]
    trn1 z15.h, z0.h, z3.h
    trn1 z16.h, z1.h, z3.h
    trn1 z17.h, z0.h, z4.h
    trn1 z18.h, z1.h, z4.h
    ldr x12, [x4]
    st1 {v15.4s-v18.4s}, [x12]
    add x12, x12, #64 // 16 elements
    str x12, [x4]

    // Branch based on q
    cmp x11, #1
    b.ne q_gt_one_p2

q_is_one_p2:
    ldrh w25, [x20, x25, lsl #1] // entropy[sig]
    dup z0.d, x25
    ldr varX_LAMBDA2, [X29, #280] // arg 24
    mov TMP_Z2.d, varX_LAMBDA2
    mul z0.d, Z0.d, TMP_Z2.d
    lsr z0.d, P_TRUE/m, Z0.d, #CABAC_SHIFT_CONST_VAL
    fmov x12, d14
    mov TMP_Z2.d, x12
    add z0.d,  z0.d, TMP_Z2.d
    sub z14.d, z14.d, z0.d

   // TRELLIS_COEF0_1_SVE_
    ldr z28, [x5] // Load lev_used0123
    str z28, [x3] // Store cur_level0123
    mov z27.d, z28.d
    add z27.s, z27.s, #4
    ldr x9, [x29, #312]
    str z27, [x9] // Store cur_level4567
    add z28.s, z28.s, #8
    str z28, [x5] // Store update back to lev_used0123

    ld1h    {z1.s}, P_TRUE/z, [x20, z7.s, uxtw #1]
    TRELLIS_COEF1_1_SVE_(1)

    b phase2_loop_start

q_gt_one_p2:
    dup z0.d, #-1
    dup CUR_SCORE02.d, #-1
    ldr CUR_SCORE02_PTR, [x29, #208]
    str z0, [CUR_SCORE02_PTR] // cur_score02 = MAX
    ldr x14, [x29, #216]
    str z0, [x14]             // cur_score13 = MAX
    ldr x14, [x29, #224]
    str z0, [x14]             // cur_score46 = MAX
    mov z5.d, z0.d            // cur_score57 = MAX
    mov z31.d, z0.d           // cur_score57 = MAX

    mov z11.d, z6.d           // prev_cabac4567 = cur_cabac4567

    eor  w18, w18, #1
    ldrh w18, [x20, x18, lsl #1]
    ldr  x16, [x29, #288]
    ldr  z3, [x16]
    mov  v3.s[3], w18
    str  z3, [x16] // *entropy012_xor

    mov z15.d, z7.d
    eor z15.s, z15.s, #1
    ld1h {z4.s}, P_TRUE/z, [x20, z15.s, uxtw #1] // entropy4567xor

    trn2 z2.h, z6.h, z6.h
    cmp var_DC, #0
    b.ne dc_no_revb_p2
    revb z2.h, P_S3/m, z2.H
dc_no_revb_p2:
    sel z2.b, P_S23, z2.b, LEVEL_STATE6767.b // lgt1_state4567

    cmp x11, #2
    b.ne q_gt_two_p2

q_is_two_p2:
    ld1h {z1.s}, P_TRUE/z, [x20, z7.s, UXTW #1] // entropy4567
    TRELLIS_COEF1_1_SVE_(0)
    lsr w24, w24, #8
    trn2 z8.b, z8.b, z8.b
    TRELLIS_COEF2_1_SVE_
    b phase2_loop_start

q_gt_two_p2:
    lsr w24, w24, #8
    trn2 z8.b, z8.b, z8.b
    sub w25, w11, #1
    TRELLIS_COEFN_1_SVE_ 0, w25
    TRELLIS_COEFN_1_SVE_ 1, w11
    b phase2_loop_start

phase2_loop_end:
    ldr CUR_SCORE02_PTR, [X29, #208] // cur_score02 (arg 15)
    ldr CUR_SCORE13_PTR, [X29, #216] // cur_score13 (arg 16)
    ldr CUR_SCORE46_PTR, [X29, #224] // cur_score46 (arg 17)

    // Load score vectors from memory into SVE registers
    ldr CUR_SCORE02, [CUR_SCORE02_PTR]  // Z3 = cur_score02
    mov CUR_SCORE02.d, CUR_SCORE02.d[1]
    ldr CUR_SCORE13, [CUR_SCORE13_PTR]  // Z4 = cur_score13
    ldr z9, [CUR_SCORE46_PTR]           // Z9 = cur_score46
    str CUR_SCORE02, [CUR_SCORE02_PTR]  // Z3 = cur_score02

    mov     z3.d, CUR_SCORE02.d         // Z3 = original cur_score02
    mov     z4.d, CUR_SCORE13.d         // Z4 = cur_score13
    mov     z6.d, z9.d                  // Z6 = cur_score46
    mov     z7.d, CUR_SCORE57.d         // Z7 = cur_score57

    // Find the single minimum score across all nodes.
    movprfx z10, z3
    umin    z10.d, P_TRUE/m, z10.d, CUR_SCORE13.d // min_score_a = min(*cur_score02, *cur_score13)
    movprfx z12, z6
    umin    z12.d, P_TRUE/m, z12.d, z7.d    // min_score_b = min(*cur_score46, cur_score57)
    umin    z10.d, P_TRUE/m, z10.d, z12.d   // min(min_score_a, min_score_b)
    uminv   d0, P_TRUE, z10.d               // Get scalar minimum value from vector
    mov     x11, v0.d[0]                    // X11 = min_score

    // Compare each node's score with the minimum to find the winner(s).
    mov TMP_Z5.d, x11
    cmpeq   p3.d, P_S2/z, z3.d, TMP_Z5.d    // P3 = j_2 (governed by s2)
    cmpeq   p4.d, P_TRUE/z, z4.d, TMP_Z5.d  // P4 = j13
    cmpeq   p5.d, P_TRUE/z, z6.d, TMP_Z5.d  // P5 = j46
    cmpeq   p6.d, P_TRUE/z, z7.d, TMP_Z5.d  // P6 = j57

    // Interleave and find the first winning node in each group to resolve ties.
    // P15 (pf) is an all-true predicate.
    trn1   p9.s, p3.s, p4.s     // Interleave j_2 and j13 predicates
    mov    p7.b, p9.b
    pfirst p7.b, p7, p7.b       // P7 = j_123 (find first true lane)
    trn1   p10.s, p5.s, p6.s    // Interleave j46 and j57 predicates
    mov    p8.b, p10.b
    pfirst p3.b, p8, P_FALSE.b       // P8 = j4567 (find first true lane)

    // Load the level vectors associated with the scores.
    ldr  CUR_LEVEL4567_PTR, [x29, #312]
    ldr  z8, [CUR_LEVEL4567_PTR]
    ldr  z9, [x3]

    lastb  w0, p3, z8.s
    clastb w0, p7, w0, z9.s
    ldr x17, [x29, #344]
    str w0, [x17]
    mov w21, w0
    mov w0, #1
    b last_loop

.phase2_else_clause:
    // Phase 1 completed without encountering any coefficient with abs(q) >= 2.
    // (i.e., all processed coefficients had abs(q) == 1, or q == 0).
    // min_score = svminv_u64( p, svmin_u64_x(p, cur_score02, cur_score13));
    ldr z0, [x25] // cur_score02
    ldr z1, [x13] // cur_scure13
    mov z2.d, z0.d
    umin z2.d, P_TRUE/m, z2.d, z1.d
    uminv d3, P_TRUE, z2.d

    // j02 = svcmpeq_n_u64( p, cur_score02, min_score );
    fmov x10, d3
    dup z3.d, x10
    cmpeq p1.d, P_TRUE/z, z0.d, z3.d

    // if(__builtin_expect(svptest_first( p, j02 ), 1))
    ptest   P_TRUE, p1.b
    b.pl .final_calculations
    mov w0, #0
// We only need to zero an empty 4x4 block. 8x8 can be implicitly emptied via zero nnz, as can dc.
// if( num_coefs == 16 && !dc )
    cmp w26, #16
    b.ne    epilogue   // if num_coefs != 16, continue
    cbz     var_DC, .Lperform_memset                  // if dc == 0, perform memset
    b       epilogue                    // else (dc != 0), continue
.Lperform_memset:
    ldr x20, [x29, #352]
#if HIGH_BIT_DEPTH
    stp     xzr, xzr, [x20, #0]
    stp     xzr, xzr, [x20, #16]
    stp     xzr, xzr, [x20, #32]
    stp     xzr, xzr, [x20, #48]
#else
    stp     xzr, xzr, [x20, #0]
    stp     xzr, xzr, [x20, #16]
#endif
    b epilogue

.final_calculations:
    // j13 = svcmpeq_n_u64( p, cur_score13, min_score );
    cmpeq p2.d, P_TRUE/z, z1.d, z3.d

    // j0123 = svpfirst_b( svtrn1_b32(j02, j13), pf );
    trn1   p1.s, p1.s, p2.s
    pfirst p3.b, p1, p3.b
    ldr z5, [x3]
    lastb w21, p3, z5.s
    ldr x17, [x29, #344]
    str w21, [x17]
last_loop:

    mov w0, #1
    ldr w22, [x29, #360] // w22 = last_nnz, w1 = b_ac
    cmp w22, #3
    b.lo epilogue
    sub w3, w22, #3      // loop end
last_loop_start:
    cmp w1, w3
    b.hi epilogue

    ld1b {z10.s}, P_TRUE/z, [x6, x1]

    mov x25, xzr
    ldr x27, [x29, #368]
    // Unrolled for( int j = 0; j < 64; j += 16 )
    // j = 0
    ldr w24, [x27, x21, lsl #2]
    ubfx x28, x24, #16, #16 // leaf >> 16
    orr x25, x25, x28
    and w21, w24, #0xffff  //level = leaf & 0xffff

    // j = 16
    ldr w24, [x27, x21, lsl #2]
    ubfx x28, x24, #16, #16 // leaf >> 16
    orr x25, x25, x28, lsl #16
    and w21, w24, #0xffff  //level = leaf & 0xffff

    // j = 32
    ldr w24, [x27, x21, lsl #2]
    ubfx x28, x24, #16, #16 // leaf >> 16
    orr x25, x25, x28, lsl #32
    and w21, w24, #0xffff  //level = leaf & 0xffff

    // j = 16
    ldr w24, [x27, x21, lsl #2]
    ubfx x28, x24, #16, #16 // leaf >> 16
    orr x25, x25, x28, lsl #48
    and w21, w24, #0xffff  //level = leaf & 0xffff

    // lev = svreinterpret_s32(svunpklo_u32(svreinterpret_u16(svdup_u64(tmp_lev))));
    dup     z4.d, x25
    uunpklo z5.s, z4.h

.if HIGH_BIT_DEPTH
    // sign = svld1_s32( p, &quant_coefs[i] );
    ld1w    { z11.s }, P_TRUE/z, [x2, x1, lsl #2]
.else
    // sign = svld1sh_s32( p, &quant_coefs[i] );
    ld1sh   { z11.s }, P_TRUE/z, [x2, x1, lsl #1]
.endif

    asr z11.s, z11.s, #31
    eor z5.s, P_TRUE/m, z5.s, z11.s
    sub z5.s, P_TRUE/m, z5.s, z11.s

    ldr x20, [x29, #352]
.if HIGH_BIT_DEPTH
    st1w {z5.s}, P_TRUE, [x20, z10.s, uxtw #2]
.else
    st1h {z5.s}, P_TRUE, [x20, z10.s, uxtw #1]
.endif
    add w1, w1, #4
    ldr x17, [x29, #344]
    str w21, [x17]
    b last_loop_start

epilogue:
    // Restore vector registers from stack
    ldp d8, d9,   [sp, #0]
    ldp d10, d11, [sp, #16]
    ldp d12, d13, [sp, #32]
    ldp d14, d15, [sp, #48]
    mov sp, x29

    // Restore callee-saved GPRs
    ldp x19, x20, [sp, #16]
    ldp x21, x22, [sp, #32]
    ldp x23, x24, [sp, #48]
    ldp x25, x26, [sp, #64]
    ldp x27, x28, [sp, #80]

    // Restore Frame Pointer and Link Register, deallocating the entire 224-byte stack frame
    ldp x29, x30, [sp], #160
    ret

