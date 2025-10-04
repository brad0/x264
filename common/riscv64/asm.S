/*****************************************************************************
 * asm.S: riscv64 utility macros
 *****************************************************************************
 * Copyright (C) 2025 x264 project
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

#ifndef ASM_S_  // #include guards
#define ASM_S_

#include "config.h"

.option arch, +v

#define GLUE(a, b) a ## b
#define JOIN(a, b) GLUE(a, b)

#ifdef PREFIX
#   define BASE _x264_
#   define SYM_PREFIX _
#else
#   define BASE x264_
#   define SYM_PREFIX
#endif

#ifdef BIT_DEPTH
#   define EXTERN_ASM JOIN(JOIN(BASE, BIT_DEPTH), _)
#else
#   define EXTERN_ASM BASE
#endif

#define X(s) JOIN(EXTERN_ASM, s)
#define X264(s) JOIN(BASE, s)
#define EXT(s) JOIN(SYM_PREFIX, s)

#ifdef __ELF__
#   define ELF
#else
#   define ELF  #
#endif

#ifdef __MACH__
#   define MACH
#else
#   define MACH #
#endif

#if HAVE_AS_FUNC
#   define FUNC
#else
#   define FUNC #
#endif

.macro  function name, export=0, align=2
    .macro endfunc
.if \export
ELF     .size   EXTERN_ASM\name, . - EXTERN_ASM\name
.else
ELF     .size   \name, . - \name
.endif
FUNC    .endfunc
        .purgem endfunc
    .endm
        .text
        .align          \align
    .if \export
        .global EXTERN_ASM\name
ELF     .type   EXTERN_ASM\name, %function
FUNC    .func   EXTERN_ASM\name
EXTERN_ASM\name:
    .else
ELF     .type   \name, %function
FUNC    .func   \name
\name:
    .endif
.endm

.macro  const   name, align=2
    .macro endconst
ELF     .size   \name, . - \name
        .purgem endconst
    .endm
ELF     .section        .rodata
MACH    .const_data
        .align          \align
\name:
.endm

.macro SUMSUB_AB   sum, sub, a, b
    vadd.vv     \sum,  \a,  \b
    vsub.vv     \diff, \a,  \b
.endm

#endif
