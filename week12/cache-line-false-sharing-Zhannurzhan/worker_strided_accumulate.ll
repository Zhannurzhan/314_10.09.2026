; ModuleID = 'worker_strided_accumulate'
source_filename = "<string>"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

@.const.worker_strided_accumulate = internal constant [26 x i8] c"worker_strided_accumulate\00"
@_ZN08NumbaEnv8__main__25worker_strided_accumulateB2v1B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedExxy = common local_unnamed_addr global ptr null
@".const.missing Environment: _ZN08NumbaEnv8__main__25worker_strided_accumulateB2v1B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedExxy" = internal constant [153 x i8] c"missing Environment: _ZN08NumbaEnv8__main__25worker_strided_accumulateB2v1B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedExxy\00"
@PyExc_TypeError = external global i8
@".const.can't unbox array from PyObject into native value.  The object maybe of a different type" = internal constant [89 x i8] c"can't unbox array from PyObject into native value.  The object maybe of a different type\00"
@PyExc_RuntimeError = external global i8

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define noundef i32 @_ZN8__main__25worker_strided_accumulateB2v1B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedExxy(ptr noalias nocapture writeonly %retptr, ptr noalias nocapture readnone %excinfo, ptr nocapture readnone %arg.buffer.0, ptr nocapture readnone %arg.buffer.1, i64 %arg.buffer.2, i64 %arg.buffer.3, ptr nocapture %arg.buffer.4, i64 %arg.buffer.5.0, i64 %arg.buffer.6.0, i64 %arg.slot, i64 %arg.iters, i64 %arg.seed) local_unnamed_addr #0 {
B0.endif:
  %.11295.not = icmp slt i64 %arg.iters, 1
  br i1 %.11295.not, label %B330, label %B118.lr.ph

B118.lr.ph:                                       ; preds = %B0.endif
  %.164 = icmp slt i64 %arg.slot, 0
  %.165 = select i1 %.164, i64 %arg.buffer.5.0, i64 0
  %0 = getelementptr i64, ptr %arg.buffer.4, i64 %arg.slot
  %xtraiter = and i64 %arg.iters, 3
  %1 = icmp ult i64 %arg.iters, 4
  br i1 %1, label %B330.loopexit.unr-lcssa, label %B118.lr.ph.new

B118.lr.ph.new:                                   ; preds = %B118.lr.ph
  %unroll_iter = and i64 %arg.iters, 9223372036854775804
  br label %B118

B118:                                             ; preds = %B118, %B118.lr.ph.new
  %.1189496 = phi i64 [ 0, %B118.lr.ph.new ], [ %.125.3, %B118 ]
  %sunkaddr = mul i64 %.165, 8
  %sunkaddr1 = getelementptr inbounds i8, ptr %0, i64 %sunkaddr
  %.180 = load i64, ptr %sunkaddr1, align 8
  %.181 = and i64 %.1189496, 60
  %.183 = lshr i64 %arg.seed, %.181
  %.185 = and i64 %.183, 1
  %.187 = lshr i64 %.180, 63
  %.189 = add nsw i64 %.187, %arg.slot
  %.190 = add i64 %.180, 1
  %.192 = add i64 %.190, %.185
  %.206 = icmp slt i64 %.189, 0
  %.207 = select i1 %.206, i64 %arg.buffer.5.0, i64 0
  %2 = getelementptr i64, ptr %arg.buffer.4, i64 %.189
  %.221 = getelementptr i64, ptr %2, i64 %.207
  store i64 %.192, ptr %.221, align 8
  %.180.1 = load i64, ptr %sunkaddr1, align 8
  %.125 = and i64 %.1189496, 60
  %.181.1 = or disjoint i64 %.125, 1
  %.183.1 = lshr i64 %arg.seed, %.181.1
  %.185.1 = and i64 %.183.1, 1
  %.187.1 = lshr i64 %.180.1, 63
  %.189.1 = add nsw i64 %.187.1, %arg.slot
  %.190.1 = add i64 %.180.1, 1
  %.192.1 = add i64 %.190.1, %.185.1
  %.206.1 = icmp slt i64 %.189.1, 0
  %.207.1 = select i1 %.206.1, i64 %arg.buffer.5.0, i64 0
  %3 = getelementptr i64, ptr %arg.buffer.4, i64 %.189.1
  %.221.1 = getelementptr i64, ptr %3, i64 %.207.1
  store i64 %.192.1, ptr %.221.1, align 8
  %.180.2 = load i64, ptr %sunkaddr1, align 8
  %.125.1 = and i64 %.1189496, 60
  %.181.2 = or disjoint i64 %.125.1, 2
  %.183.2 = lshr i64 %arg.seed, %.181.2
  %.185.2 = and i64 %.183.2, 1
  %.187.2 = lshr i64 %.180.2, 63
  %.189.2 = add nsw i64 %.187.2, %arg.slot
  %.190.2 = add i64 %.180.2, 1
  %.192.2 = add i64 %.190.2, %.185.2
  %.206.2 = icmp slt i64 %.189.2, 0
  %.207.2 = select i1 %.206.2, i64 %arg.buffer.5.0, i64 0
  %4 = getelementptr i64, ptr %arg.buffer.4, i64 %.189.2
  %.221.2 = getelementptr i64, ptr %4, i64 %.207.2
  store i64 %.192.2, ptr %.221.2, align 8
  %.125.3 = add nuw nsw i64 %.1189496, 4
  %.180.3 = load i64, ptr %sunkaddr1, align 8
  %.125.2 = and i64 %.1189496, 60
  %.181.3 = or disjoint i64 %.125.2, 3
  %.183.3 = lshr i64 %arg.seed, %.181.3
  %.185.3 = and i64 %.183.3, 1
  %.187.3 = lshr i64 %.180.3, 63
  %.189.3 = add nsw i64 %.187.3, %arg.slot
  %.190.3 = add i64 %.180.3, 1
  %.192.3 = add i64 %.190.3, %.185.3
  %.206.3 = icmp slt i64 %.189.3, 0
  %.207.3 = select i1 %.206.3, i64 %arg.buffer.5.0, i64 0
  %5 = getelementptr i64, ptr %arg.buffer.4, i64 %.189.3
  %.221.3 = getelementptr i64, ptr %5, i64 %.207.3
  store i64 %.192.3, ptr %.221.3, align 8
  %niter.ncmp.3 = icmp eq i64 %unroll_iter, %.125.3
  br i1 %niter.ncmp.3, label %B330.loopexit.unr-lcssa, label %B118

B330.loopexit.unr-lcssa:                          ; preds = %B118, %B118.lr.ph
  %.1189496.unr = phi i64 [ 0, %B118.lr.ph ], [ %.125.3, %B118 ]
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod.not, label %B330, label %B118.epil.preheader

B118.epil.preheader:                              ; preds = %B330.loopexit.unr-lcssa
  br label %B118.epil

B118.epil:                                        ; preds = %B118.epil.preheader, %B118.epil
  %lsr.iv = phi i64 [ %xtraiter, %B118.epil.preheader ], [ %lsr.iv.next, %B118.epil ]
  %.1189496.epil = phi i64 [ %.125.epil, %B118.epil ], [ %.1189496.unr, %B118.epil.preheader ]
  %.125.epil = add nuw nsw i64 %.1189496.epil, 1
  %sunkaddr2 = mul i64 %.165, 8
  %sunkaddr3 = getelementptr inbounds i8, ptr %0, i64 %sunkaddr2
  %.180.epil = load i64, ptr %sunkaddr3, align 8
  %.181.epil = and i64 %.1189496.epil, 63
  %.183.epil = lshr i64 %arg.seed, %.181.epil
  %.185.epil = and i64 %.183.epil, 1
  %.187.epil = lshr i64 %.180.epil, 63
  %.189.epil = add nsw i64 %.187.epil, %arg.slot
  %.190.epil = add i64 %.180.epil, 1
  %.192.epil = add i64 %.190.epil, %.185.epil
  %.206.epil = icmp slt i64 %.189.epil, 0
  %.207.epil = select i1 %.206.epil, i64 %arg.buffer.5.0, i64 0
  %6 = getelementptr i64, ptr %arg.buffer.4, i64 %.189.epil
  %.221.epil = getelementptr i64, ptr %6, i64 %.207.epil
  store i64 %.192.epil, ptr %.221.epil, align 8
  %lsr.iv.next = add nsw i64 %lsr.iv, -1
  %epil.iter.cmp.not = icmp eq i64 %lsr.iv.next, 0
  br i1 %epil.iter.cmp.not, label %B330, label %B118.epil, !llvm.loop !0

B330:                                             ; preds = %B118.epil, %B330.loopexit.unr-lcssa, %B0.endif
  %.248 = icmp slt i64 %arg.slot, 0
  %.249 = select i1 %.248, i64 %arg.buffer.5.0, i64 0
  %7 = getelementptr i64, ptr %arg.buffer.4, i64 %arg.slot
  %.263 = getelementptr i64, ptr %7, i64 %.249
  %.264 = load i64, ptr %.263, align 8
  store i64 %.264, ptr %retptr, align 8
  ret i32 0
}

define ptr @_ZN7cpython8__main__25worker_strided_accumulateB2v1B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedExxy(ptr nocapture readnone %py_closure, ptr %py_args, ptr nocapture readnone %py_kws) local_unnamed_addr {
entry:
  %.5 = alloca ptr, align 8
  %.6 = alloca ptr, align 8
  %.7 = alloca ptr, align 8
  %.8 = alloca ptr, align 8
  %.9 = call i32 (ptr, ptr, i64, i64, ...) @PyArg_UnpackTuple(ptr %py_args, ptr nonnull @.const.worker_strided_accumulate, i64 4, i64 4, ptr nonnull %.5, ptr nonnull %.6, ptr nonnull %.7, ptr nonnull %.8)
  %.10 = icmp eq i32 %.9, 0
  %.24 = alloca { ptr, ptr, i64, i64, ptr, [1 x i64], [1 x i64] }, align 8
  %.95 = alloca i64, align 8
  store i64 0, ptr %.95, align 8
  br i1 %.10, label %common.ret, label %entry.endif, !prof !2

common.ret:                                       ; preds = %entry.endif.endif.endif.thread, %arg0.err, %entry, %entry.endif.endif.endif.e...endif.endif, %entry.endif.if
  %common.ret.op = phi ptr [ null, %entry.endif.if ], [ %.122, %entry.endif.endif.endif.e...endif.endif ], [ null, %entry ], [ null, %arg0.err ], [ null, %entry.endif.endif.endif.thread ]
  ret ptr %common.ret.op

entry.endif:                                      ; preds = %entry
  %.14 = load ptr, ptr @_ZN08NumbaEnv8__main__25worker_strided_accumulateB2v1B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedExxy, align 8
  %.19 = icmp eq ptr %.14, null
  br i1 %.19, label %entry.endif.if, label %entry.endif.endif, !prof !2

entry.endif.if:                                   ; preds = %entry.endif
  call void @PyErr_SetString(ptr nonnull @PyExc_RuntimeError, ptr nonnull @".const.missing Environment: _ZN08NumbaEnv8__main__25worker_strided_accumulateB2v1B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedExxy")
  br label %common.ret

entry.endif.endif:                                ; preds = %entry.endif
  %.23 = load ptr, ptr %.5, align 8
  call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(56) %.24, i8 0, i64 56, i1 false)
  %.28 = call i32 @NRT_adapt_ndarray_from_python(ptr %.23, ptr nonnull %.24)
  %sunkaddr = getelementptr inbounds i8, ptr %.24, i64 24
  %.32 = load i64, ptr %sunkaddr, align 8
  %.33 = icmp ne i64 %.32, 8
  %.34 = icmp ne i32 %.28, 0
  %.35 = or i1 %.34, %.33
  br i1 %.35, label %entry.endif.endif.endif.thread, label %entry.endif.endif.endif.endif, !prof !2

entry.endif.endif.endif.thread:                   ; preds = %entry.endif.endif
  call void @PyErr_SetString(ptr nonnull @PyExc_TypeError, ptr nonnull @".const.can't unbox array from PyObject into native value.  The object maybe of a different type")
  br label %common.ret

entry.endif.endif.endif.endif:                    ; preds = %entry.endif.endif
  %.39.fca.0.load = load ptr, ptr %.24, align 8
  %sunkaddr1 = getelementptr inbounds i8, ptr %.24, i64 32
  %.39.fca.4.load = load ptr, ptr %sunkaddr1, align 8
  %sunkaddr2 = getelementptr inbounds i8, ptr %.24, i64 40
  %.39.fca.5.0.load = load i64, ptr %sunkaddr2, align 8
  %.46 = load ptr, ptr %.6, align 8
  %.49 = call ptr @PyNumber_Long(ptr %.46)
  %.50.not = icmp eq ptr %.49, null
  br i1 %.50.not, label %entry.endif.endif.endif.endif.endif, label %entry.endif.endif.endif.endif.if, !prof !2

arg0.err:                                         ; preds = %entry.endif.endif.endif.e...endif, %entry.endif.endif.endif.endif.endif.endif.endif, %entry.endif.endif.endif.endif.endif
  call void @NRT_decref(ptr %.39.fca.0.load)
  br label %common.ret

entry.endif.endif.endif.endif.if:                 ; preds = %entry.endif.endif.endif.endif
  %.52 = call i64 @PyLong_AsLongLong(ptr nonnull %.49)
  call void @Py_DecRef(ptr nonnull %.49)
  br label %entry.endif.endif.endif.endif.endif

entry.endif.endif.endif.endif.endif:              ; preds = %entry.endif.endif.endif.endif.if, %entry.endif.endif.endif.endif
  %.47.0 = phi i64 [ %.52, %entry.endif.endif.endif.endif.if ], [ 0, %entry.endif.endif.endif.endif ]
  %.57 = call ptr @PyErr_Occurred()
  %.58.not = icmp eq ptr %.57, null
  br i1 %.58.not, label %entry.endif.endif.endif.endif.endif.endif, label %arg0.err, !prof !3

entry.endif.endif.endif.endif.endif.endif:        ; preds = %entry.endif.endif.endif.endif.endif
  %.62 = load ptr, ptr %.7, align 8
  %.65 = call ptr @PyNumber_Long(ptr %.62)
  %.66.not = icmp eq ptr %.65, null
  br i1 %.66.not, label %entry.endif.endif.endif.endif.endif.endif.endif, label %entry.endif.endif.endif.endif.endif.endif.if, !prof !2

entry.endif.endif.endif.endif.endif.endif.if:     ; preds = %entry.endif.endif.endif.endif.endif.endif
  %.68 = call i64 @PyLong_AsLongLong(ptr nonnull %.65)
  call void @Py_DecRef(ptr nonnull %.65)
  br label %entry.endif.endif.endif.endif.endif.endif.endif

entry.endif.endif.endif.endif.endif.endif.endif:  ; preds = %entry.endif.endif.endif.endif.endif.endif.if, %entry.endif.endif.endif.endif.endif.endif
  %.63.0 = phi i64 [ %.68, %entry.endif.endif.endif.endif.endif.endif.if ], [ 0, %entry.endif.endif.endif.endif.endif.endif ]
  %.73 = call ptr @PyErr_Occurred()
  %.74.not = icmp eq ptr %.73, null
  br i1 %.74.not, label %entry.endif.endif.endif.endif.endif.endif.endif.endif, label %arg0.err, !prof !3

entry.endif.endif.endif.endif.endif.endif.endif.endif: ; preds = %entry.endif.endif.endif.endif.endif.endif.endif
  %.78 = load ptr, ptr %.8, align 8
  %.81 = call ptr @PyNumber_Long(ptr %.78)
  %.82.not = icmp eq ptr %.81, null
  br i1 %.82.not, label %entry.endif.endif.endif.e...endif, label %entry.endif.endif.endif.e...if, !prof !2

entry.endif.endif.endif.e...if:                   ; preds = %entry.endif.endif.endif.endif.endif.endif.endif.endif
  %.84 = call i64 @PyLong_AsUnsignedLongLong(ptr nonnull %.81)
  call void @Py_DecRef(ptr nonnull %.81)
  br label %entry.endif.endif.endif.e...endif

entry.endif.endif.endif.e...endif:                ; preds = %entry.endif.endif.endif.e...if, %entry.endif.endif.endif.endif.endif.endif.endif.endif
  %.79.0 = phi i64 [ %.84, %entry.endif.endif.endif.e...if ], [ 0, %entry.endif.endif.endif.endif.endif.endif.endif.endif ]
  %.89 = call ptr @PyErr_Occurred()
  %.90.not = icmp eq ptr %.89, null
  br i1 %.90.not, label %entry.endif.endif.endif.e...endif.endif, label %arg0.err, !prof !3

entry.endif.endif.endif.e...endif.endif:          ; preds = %entry.endif.endif.endif.e...endif
  %.94 = call ptr @PyEval_SaveThread()
  store i64 0, ptr %.95, align 8
  %.101 = call i32 @_ZN8__main__25worker_strided_accumulateB2v1B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedExxy(ptr nonnull %.95, ptr nonnull poison, ptr poison, ptr poison, i64 poison, i64 poison, ptr %.39.fca.4.load, i64 %.39.fca.5.0.load, i64 poison, i64 %.47.0, i64 %.63.0, i64 %.79.0) #3
  %.111 = load i64, ptr %.95, align 8
  call void @PyEval_RestoreThread(ptr %.94)
  call void @NRT_decref(ptr %.39.fca.0.load)
  %.122 = call ptr @PyLong_FromUnsignedLongLong(i64 %.111)
  br label %common.ret
}

declare i32 @PyArg_UnpackTuple(ptr, ptr, i64, i64, ...) local_unnamed_addr

declare void @PyErr_SetString(ptr, ptr) local_unnamed_addr

; Function Attrs: mustprogress nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #1

declare i32 @NRT_adapt_ndarray_from_python(ptr nocapture, ptr nocapture) local_unnamed_addr

declare ptr @PyNumber_Long(ptr) local_unnamed_addr

declare i64 @PyLong_AsLongLong(ptr) local_unnamed_addr

declare void @Py_DecRef(ptr) local_unnamed_addr

declare ptr @PyErr_Occurred() local_unnamed_addr

declare i64 @PyLong_AsUnsignedLongLong(ptr) local_unnamed_addr

declare ptr @PyEval_SaveThread() local_unnamed_addr

declare void @PyEval_RestoreThread(ptr) local_unnamed_addr

declare ptr @PyLong_FromUnsignedLongLong(i64) local_unnamed_addr

; Function Attrs: nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none)
define i64 @cfunc._ZN8__main__25worker_strided_accumulateB2v1B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedExxy({ ptr, ptr, i64, i64, ptr, [1 x i64], [1 x i64] } %.1, i64 %.2, i64 %.3, i64 %.4) local_unnamed_addr #2 {
entry:
  %.6 = alloca i64, align 8
  store i64 0, ptr %.6, align 8
  %extracted.data = extractvalue { ptr, ptr, i64, i64, ptr, [1 x i64], [1 x i64] } %.1, 4
  %extracted.shape = extractvalue { ptr, ptr, i64, i64, ptr, [1 x i64], [1 x i64] } %.1, 5
  %.10 = extractvalue [1 x i64] %extracted.shape, 0
  %.12 = call i32 @_ZN8__main__25worker_strided_accumulateB2v1B40c8tJTIeFIjxB2IKSgI4CrvQClQZ6FcpCShpgEU0AE5ArrayIyLi1E1C7mutable7alignedExxy(ptr nonnull %.6, ptr nonnull poison, ptr poison, ptr poison, i64 poison, i64 poison, ptr %extracted.data, i64 %.10, i64 poison, i64 %.2, i64 %.3, i64 %.4) #3
  %.22 = load i64, ptr %.6, align 8
  ret i64 %.22
}

; Function Attrs: noinline
define linkonce_odr void @NRT_decref(ptr %.1) local_unnamed_addr #3 {
.3:
  %.4 = icmp eq ptr %.1, null
  br i1 %.4, label %common.ret1, label %.3.endif, !prof !2

common.ret1:                                      ; preds = %.3, %.3.endif
  ret void

.3.endif:                                         ; preds = %.3
  fence release
  %0 = tail call i8 @llvm.x86.atomic.sub.cc.i64(ptr nonnull %.1, i64 1, i32 4)
  %1 = trunc i8 %0 to i1
  br i1 %1, label %.3.endif.if, label %common.ret1, !prof !2

.3.endif.if:                                      ; preds = %.3.endif
  fence acquire
  tail call void @NRT_MemInfo_call_dtor(ptr nonnull %.1)
  ret void
}

; Function Attrs: nounwind
declare i8 @llvm.x86.atomic.sub.cc.i64(ptr, i64, i32 immarg) #4

declare void @NRT_MemInfo_call_dtor(ptr) local_unnamed_addr

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) }
attributes #1 = { mustprogress nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #2 = { nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none) }
attributes #3 = { noinline }
attributes #4 = { nounwind }

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
!2 = !{!"branch_weights", i32 1, i32 99}
!3 = !{!"branch_weights", i32 99, i32 1}
