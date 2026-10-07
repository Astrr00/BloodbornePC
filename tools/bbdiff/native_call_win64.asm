; SPDX-License-Identifier: GPL-3.0-or-later
; bb_native_call(NativeState* s) -> 1: run guest code natively with a complete guest register file.
; NativeState layout (tools/bbdiff/main.cpp): r[16] @0 (x86 order, rsp = r[4] @32), rflags @128,
; xmm[16][16] @136, host_rsp @392, fn @400. Not reentrant (uses globals to survive the register swap).
; bb_native_recover: target for the vectored exception handler when guest code faults (guest frames have no unwind
; data, so SEH cannot reach the caller); restores the host frame from host_rsp and returns 0.

.data
g_state QWORD 0
g_fn    QWORD 0
g_rax   QWORD 0

PUBLIC bb_native_after_call
.code
bb_native_call PROC
    push rbx
    push rbp
    push rdi
    push rsi
    push r12
    push r13
    push r14
    push r15
    sub rsp, 168                    ; xmm6-15 save area (keeps 16-byte alignment)
    movdqu [rsp+0], xmm6
    movdqu [rsp+16], xmm7
    movdqu [rsp+32], xmm8
    movdqu [rsp+48], xmm9
    movdqu [rsp+64], xmm10
    movdqu [rsp+80], xmm11
    movdqu [rsp+96], xmm12
    movdqu [rsp+112], xmm13
    movdqu [rsp+128], xmm14
    movdqu [rsp+144], xmm15
    mov [rcx+392], rsp
    mov g_state, rcx
    mov rax, [rcx+400]
    mov g_fn, rax

    movdqu xmm0, [rcx+136]
    movdqu xmm1, [rcx+152]
    movdqu xmm2, [rcx+168]
    movdqu xmm3, [rcx+184]
    movdqu xmm4, [rcx+200]
    movdqu xmm5, [rcx+216]
    movdqu xmm6, [rcx+232]
    movdqu xmm7, [rcx+248]
    movdqu xmm8, [rcx+264]
    movdqu xmm9, [rcx+280]
    movdqu xmm10, [rcx+296]
    movdqu xmm11, [rcx+312]
    movdqu xmm12, [rcx+328]
    movdqu xmm13, [rcx+344]
    movdqu xmm14, [rcx+360]
    movdqu xmm15, [rcx+376]

    mov rax, [rcx+0]
    mov rdx, [rcx+16]
    mov rbx, [rcx+24]
    mov rbp, [rcx+40]
    mov rsi, [rcx+48]
    mov rdi, [rcx+56]
    mov r8, [rcx+64]
    mov r9, [rcx+72]
    mov r10, [rcx+80]
    mov r11, [rcx+88]
    mov r12, [rcx+96]
    mov r13, [rcx+104]
    mov r14, [rcx+112]
    mov r15, [rcx+120]
    mov rsp, [rcx+32]
    mov rcx, [rcx+8]
    call qword ptr [g_fn]
bb_native_after_call LABEL NEAR      ; the return address the guest function sees on its stack

    mov g_rax, rax
    mov rax, g_state
    mov [rax+8], rcx
    mov [rax+16], rdx
    mov [rax+24], rbx
    mov [rax+32], rsp
    mov [rax+40], rbp
    mov [rax+48], rsi
    mov [rax+56], rdi
    mov [rax+64], r8
    mov [rax+72], r9
    mov [rax+80], r10
    mov [rax+88], r11
    mov [rax+96], r12
    mov [rax+104], r13
    mov [rax+112], r14
    mov [rax+120], r15
    pushfq                          ; on the guest stack, below the final guest rsp
    pop rcx
    mov [rax+128], rcx
    mov rcx, g_rax
    mov [rax+0], rcx
    movdqu [rax+136], xmm0
    movdqu [rax+152], xmm1
    movdqu [rax+168], xmm2
    movdqu [rax+184], xmm3
    movdqu [rax+200], xmm4
    movdqu [rax+216], xmm5
    movdqu [rax+232], xmm6
    movdqu [rax+248], xmm7
    movdqu [rax+264], xmm8
    movdqu [rax+280], xmm9
    movdqu [rax+296], xmm10
    movdqu [rax+312], xmm11
    movdqu [rax+328], xmm12
    movdqu [rax+344], xmm13
    movdqu [rax+360], xmm14
    movdqu [rax+376], xmm15

    mov rsp, [rax+392]
    movdqu xmm6, [rsp+0]
    movdqu xmm7, [rsp+16]
    movdqu xmm8, [rsp+32]
    movdqu xmm9, [rsp+48]
    movdqu xmm10, [rsp+64]
    movdqu xmm11, [rsp+80]
    movdqu xmm12, [rsp+96]
    movdqu xmm13, [rsp+112]
    movdqu xmm14, [rsp+128]
    movdqu xmm15, [rsp+144]
    add rsp, 168
    pop r15
    pop r14
    pop r13
    pop r12
    pop rsi
    pop rdi
    pop rbp
    pop rbx
    mov eax, 1
    ret
bb_native_call ENDP

bb_native_recover PROC
    mov rax, g_state
    mov rsp, [rax+392]
    movdqu xmm6, [rsp+0]
    movdqu xmm7, [rsp+16]
    movdqu xmm8, [rsp+32]
    movdqu xmm9, [rsp+48]
    movdqu xmm10, [rsp+64]
    movdqu xmm11, [rsp+80]
    movdqu xmm12, [rsp+96]
    movdqu xmm13, [rsp+112]
    movdqu xmm14, [rsp+128]
    movdqu xmm15, [rsp+144]
    add rsp, 168
    pop r15
    pop r14
    pop r13
    pop r12
    pop rsi
    pop rdi
    pop rbp
    pop rbx
    xor eax, eax
    ret
bb_native_recover ENDP
END
