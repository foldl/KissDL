#include <stdint.h>
#include <math.h>

#define CLIP(v, a, b) ((v) < (a) ? (a) : ((v) > (b) ? b : v))

#define HWD_OFFSET(shape, h, w, d) (h * shape[1] * shape[2] + w * shape[2] + d)
#define NHWD_OFFSET(shape, n, h, w, d) (n * shape[1] * shape[2] * shape[3] + h * shape[2] * shape[3] + w * shape[3] + d)

#ifdef __GNUC__

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })

#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

#else

#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

#endif

#define _ATTRIBUTE_ __attribute__((weak))

void model_run(void);

// fixed point (Q31) multiplication with saturation
int32_t q31_mult_sat(int32_t a, int32_t b)
{
    if ((a == b) && (a = 0x80000000))
        return 0x80000000;
    else
    {
        int64_t t = a * b;
        int64_t rounding = t >= 0 ? (1 << 30) : (1 - (1 << 30));
        return (t + rounding) >> 31;
    }
}

// fixed point (Q15) multiplication with saturation
int16_t q15_mult_sat(int16_t a, int16_t b)
{
    if ((a == b) && (a = 0x8000))
        return 0x8000;
    else
    {
        int32_t t = a * b;
        int32_t rounding = t >= 0 ? (1 << 14) : (1 - (1 << 14));
        return (t + rounding) >> 15;
    }
}

// shift must be > 0
int16_t i16_round_shift_right(int16_t v, int shift)
{
    int16_t rounding;
    int16_t r;
    rounding = v & (1 << (shift - 1));
    r = v >> shift;
    if (rounding) r++;
    return r;
}

// shift must be > 0
int32_t i32_round_shift_right(int32_t v, int shift)
{
    int32_t rounding;
    int32_t r;
    rounding = v & (1 << (shift - 1));
    r = v >> shift;
    if (rounding) r++;
    return r;
}
