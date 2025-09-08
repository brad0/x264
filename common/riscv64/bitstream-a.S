/*****************************************************************************
 * bitstream-a.S: riscv64 bitstream functions
 *****************************************************************************
 * Copyright (C) 2014-2025 x264 project
 *
 * Authors: Changsheng Wu <wu.changsheng@sanechips.com.cn>
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

// uint8_t *nal_escape_c( uint8_t *dst, uint8_t *src, uint8_t *end )
function nal_escape_rvv, export=1
    sub             a3, a2, a1
    li              t1, 2
    ble             a3, t1, finish2
    vsetvli         t1, zero, e8, m2, ta, ma
    addi            a4, t1, -2
    addi            a7, t1, -1
    vmv.v.i         v2, 4
    vmv.v.i         v4, 4
    li              a5, 0xffffff
    li              a6, 3

loop:
    vsetvli         t1, a3, e8, m2, ta, ma
    vle8.v          v12, (a1)
    add             a1, a1, t1
    sub             a3, a3, t1
    vslideup.vi     v2, v12, 2
    vslideup.vi     v4, v12, 1
    vmsleu.vi       v6, v12, 3
    vmseq.vi        v8, v2, 0
    vmseq.vi        v10, v4, 0
    vmand.mm        v6, v6, v8
    vmand.mm        v6, v6, v10
    vfirst.m        t2, v6
    bgez            t2, insert1
    vse8.v          v12, (a0)
    add             a0, a0, t1
    beqz            a3, finish
    vslidedown.vx   v2, v12, a4
    vslidedown.vx   v4, v12, a7
    j               loop

insert1:
    vmsbf.m         v0, v6
    vse8.v          v12, (a0), v0.t
    add             a0, a0, t2
    vslidedown.vx   v6, v4, t2
    vslidedown.vx   v8, v2, t2
    vmv.x.s         t3, v6
    vmv.x.s         t4, v8
    andi            t4, t4, 0xff
    slli            t3, t3, 8
    or              t3, t3, t4
insert2:
    vslidedown.vx   v6, v12, t2
    vmv.x.s         t4, v6
    andi            t4, t4, 0xff
    slli            t3, t3, 8
    or              t3, t3, t4
    addi            t2, t2, 1
    and             t5, t3, a5
    bgtu            t5, a6, insert3
    sb              a6, (a0)
    addi            a0, a0, 1
    slli            t3, a6, 8
    or              t3, t3, t4
insert3:
    sb              t4, (a0)
    addi            a0, a0, 1
    blt             t2, t1, insert2
 
    vsetivli        zero, 2, e16, m1, ta, ma
    slli            t1, t4, 8
    srli            t3, t3, 8
    andi            t3, t3, 0xff
    or              t3, t3, t1
    vmv.s.x         v2, t3
    vmv.s.x         v4, t4
    bgtz            a3, loop
finish:
    ret
finish2:
    beqz            a3, finish
    vsetvli         a4, a3, e8, m1, ta, ma
    vle8.v          v2, (a1)
    vse8.v          v2, (a0)
    add             a0, a0, a4
    ret
endfunc
