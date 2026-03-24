## AscendC API Quick Reference

**Unary**: `Exp(d,s,n)` `Ln(d,s,n)` `Sqrt(d,s,n)` `Rsqrt(d,s,n)` `Abs(d,s,n)` `Reciprocal(d,s,n)` `Relu(d,s,n)`
**Binary**: `Add(d,a,b,n)` `Sub(d,a,b,n)` `Mul(d,a,b,n)` `Div(d,a,b,n)` `Max(d,a,b,n)` `Min(d,a,b,n)`
**Scalar-Vec**: `Adds(d,s,scalar,n)` `Muls(d,s,scalar,n)` `Maxs(d,s,scalar,n)` `Mins(d,s,scalar,n)` `LeakyRelu(d,s,alpha,n)`
**Compare**: `Compare(mask,a,b,mode,n)` `CompareScalar(mask,s,scalar,mode,n)` — mask type: `LocalTensor<uint8_t>`
**Select**: `Select(d,mask,a,b,SELMODE::VSEL_TENSOR_TENSOR_MODE,n)`
**Reduce**: `ReduceSum(d,s,work,n)` `ReduceMax(d,s,work,n)` `ReduceMin(d,s,work,n)` — work buffer required
**Queue**: `QuePosition::VECIN` `QuePosition::VECOUT` `QuePosition::VECCALC` (for temp buffers)
**High-level** (need #include): `Tanh` `Sigmoid` `Gelu` `Swish` — see `lib/activation/*.h`, `lib/math/*.h`

**Non-existent**: No `Neg` `Subs` `Divs` `Pow` — use `Muls(d,s,-1,n)` `Adds(d,s,-x,n)` `Muls(d,s,1/x,n)`
