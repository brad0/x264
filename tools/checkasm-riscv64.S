/****************************************************************************
 * checkasm-riscv64.S: assembly check tool
 *****************************************************************************
 * Copyright (C) 2015-2025 x264 project
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

#include "../common/riscv64/asm.S"

const register_init, align=4
.quad 0x21f86d66c8ca00ce
.quad 0x75b6ba21077c48ad
.quad 0xed56bb2dcb3c7736
.quad 0x8bda43d3fd1a7e06
.quad 0xb64a9c9e5d318408
.quad 0xdf9a54b303f1d3a3
.quad 0x4a75479abd64e097
.quad 0x249214109d5d1c88
.quad 0x1a1b2550a612b48c
.quad 0x79445c159ce79064
.quad 0x2eed899d5a28ddcd
.quad 0x86b2536fcd8cf636
endconst


const error_message
.asciz "failed to preserve register"
endconst

.text

// max number of args used by any x264 asm function.
#define MAX_ARGS 15

#define CLOBBER_STACK ((8*MAX_ARGS + 15) & ~15)

function checkasm_stack_clobber, export=1
    mv          t3, sp
    li          t2, CLOBBER_STACK
loop:
    sd          a0, (sp)
    sd          a1, 8(sp)
    addi        sp, sp, -16
    addi        t2, t2, -16
    bgtz        t2, loop
    mv          sp,  t3
    ret
endfunc

#define ARG_STACK ((8*(MAX_ARGS - 8) + 15) & ~15)

.macro push_reg op, reg1, reg2, reg3, reg4
    addi        sp, sp, -32
    \op         \reg4, (sp)
    \op         \reg3, 8(sp)
    \op         \reg2, 16(sp)
    \op         \reg1, 24(sp)
.endm

.macro pop_reg op, base, reg1, reg2, reg3, reg4
    \op         \reg1, (\base)
    \op         \reg2, 8(\base)
    \op         \reg3, 16(\base)
    \op         \reg4, 24(\base)
    addi        \base, \base, 32
.endm

function checkasm_call, export=1
    addi        sp, sp, -16
    sd          ra, (sp)
    mv          t6, sp
    push_reg    sd, s0, s1, s2, s3
    push_reg    sd, s4, s5, s6, s7
    push_reg    sd, s8, s9, s10, s11
    la          t1, register_init
    pop_reg     ld, t1, s0, s1, s2, s3
    pop_reg     ld, t1, s4, s5, s6, s7
    pop_reg     ld, t1, s8, s9, s10, s11
    addi        sp, sp, -16
    sd          a1, (sp)

    addi        sp, sp, -ARG_STACK
.equ pos, 0
.rept MAX_ARGS-8
    // Skip the first 8 args, that are loaded into registers
    addi        t2, t6, 80 + pos
    ld          t3, (t2)
    addi        t4, sp, pos
    sd          t3, (t4)
.equ pos, pos + 8
.endr
    mv          t3, a0
    addi        t2, t6, 16
    pop_reg     ld, t2, a0, a1, a2, a3
    pop_reg     ld, t2, a4, a5, a6, a7

    jalr        t3
    
    add         sp, sp, ARG_STACK
    ld          t1, (sp)
    sd          a0, (sp)
    addi        t2, sp, 8
    sd          a1, (t2)
    la          t2, register_init
    li          t5, 0

.macro check_reg reg1, reg2
    ld          t3, (t2)
    addi        t2, t2, 8
    xor         t4, t3, \reg1
    or          t5, t5, t4
    ld          t3, (t2)
    addi        t2, t2, 8
    xor         t4, t3, \reg2
    or          t5, t5, t4
.endm
    check_reg   s0, s1
    check_reg   s2, s3
    check_reg   s4, s5
    check_reg   s6, s7
    check_reg   s8, s9
    check_reg   s10, s11

    beqz        t5, finish
    li          t2, 0
    sd          t2, (t1)
    la          a0, error_message
    jal         ra, puts
finish:
    ld          a0, (sp)
    ld          a1, 8(sp)
    addi        sp, sp, 16

    pop_reg     ld, sp, s11, s10, s9, s8
    pop_reg     ld, sp, s7, s6, s5, s4
    pop_reg     ld, sp, s3, s2, s1, s0
    ld          ra, (sp)
    addi        sp, sp, 16
    ret
endfunc
