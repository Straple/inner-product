#include <bits/stdc++.h>
using namespace std;

// #pragma GCC target("avx2")

#include <emmintrin.h>
#include <tmmintrin.h>
#include <x86intrin.h>

static inline __m128i multiply128(const __m128i &a, const __m128i &b) {
    //__m128i a13    = _mm_shuffle_epi32(a, 0xF5);          // (-,a3,-,a1)
    //__m128i b13    = _mm_shuffle_epi32(b, 0xF5);          // (-,b3,-,b1)
    //__m128i prod02 = _mm_mul_epu32(a, b);                 // (-,a2*b2,-,a0*b0)
    //__m128i prod13 = _mm_mul_epu32(a13, b13);             // (-,a3*b3,-,a1*b1)
    //__m128i prod01 = _mm_unpacklo_epi32(prod02,prod13);   // (-,-,a1*b1,a0*b0)
    //__m128i prod23 = _mm_unpackhi_epi32(prod02,prod13);   // (-,-,a3*b3,a2*b2)
    //__m128i prod   = _mm_unpacklo_epi64(prod01,prod23);   // (ab3,ab2,ab1,ab0)
    // return prod;

    // SSE 4.1
    return _mm_mullo_epi32(a, b);

    //__m128i tmp1 = _mm_mul_epu32(a, b); /* mul 2,0*/
    //__m128i tmp2 =
    //    _mm_mul_epu32(_mm_srli_si128(a, 4), _mm_srli_si128(b, 4)); /* mul 3,1
    //    */
    // return _mm_unpacklo_epi32(
    //    _mm_shuffle_epi32(tmp1, _MM_SHUFFLE(0, 0, 2, 0)),
    //    _mm_shuffle_epi32(tmp2, _MM_SHUFFLE(0, 0, 2, 0))
    //); /* shuffle results to [63..0] and pack */
}

uint32_t my_inner_product(const uint32_t *a, const uint32_t *b, int len) {
    // const __m128i zero128 = _mm_setzero_si128();

    uint32_t result = 0;

    while (len % 8 != 0) {
        len--;
        result += a[len] * b[len];
    }

    uint32_t data[8];

    for (int index = 0; index < len; index += 8) {
        auto *a256 = reinterpret_cast<const __m256i *>(a + index);
        auto *b256 = reinterpret_cast<const __m256i *>(b + index);

        // std_time: 1.08776
        // my_time: 1.13766
        /*const __m128i a_lo32 = _mm_unpacklo_epi32(*a128, zero128);
        const __m128i a_hi32 = _mm_unpackhi_epi32(*a128, zero128);

        const __m128i b_lo32 = _mm_unpacklo_epi32(*b128, zero128);
        const __m128i b_hi32 = _mm_unpackhi_epi32(*b128, zero128);

        const __m128i mul_lo32 = _mm_mul_epu32(a_lo32, b_lo32);
        const __m128i mul_hi32 = _mm_mul_epu32(a_hi32, b_hi32);

        const __m128i sum = _mm_add_epi32(mul_lo32, mul_hi32);

        auto *data = reinterpret_cast<const uint32_t *>(&sum);

        result += data[0] + data[2];*/

        auto mult = _mm256_mullo_epi32(*a256, *b256);
        _mm256_storeu_epi32(data, mult);
        for (int i = 0; i < 8; i++) {
            result += data[i];
        }

        /*result += data[0];
        result += data[1];
        result += data[2];
        result += data[3];
        result += data[4];
        result += data[5];
        result += data[6];
        result += data[7];*/
    }

    return result;
}

double std_time = 0;
double my_time = 0;
double build_time = 0;

void do_test() {
    static mt19937 rnd(42);
    static vector<uint32_t> a, b;

    int size = rnd() % 100'000'000;

    {
        auto start = std::chrono::steady_clock::now();
        a.resize(size);
        b.resize(size);
        for (int i = 0; i < size; i++) {
            a[i] = rnd();
            b[i] = rnd();
        }
        build_time += std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now() - start
                      )
                          .count() /
                      1e9;
    }

    uint32_t my_ans;
    {
        auto start = std::chrono::steady_clock::now();
        my_ans = my_inner_product(a.data(), b.data(), a.size());

        my_time += std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now() - start
                   )
                       .count() /
                   1e9;
    }

    uint32_t correct_ans;
    {
        auto start = std::chrono::steady_clock::now();

        correct_ans = inner_product(
            a.begin(), a.end(), b.begin(), static_cast<uint32_t>(0)
        );

        std_time += std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - start
                    )
                        .count() /
                    1e9;
    }

    if (my_ans != correct_ans) {
        cerr << "FATAL\n";
        cerr << "correct_ans: " << correct_ans << '\n';
        cerr << "my_ans: " << my_ans << '\n';
        exit(1);
    }
}

void testing() {
    auto start = std::chrono::steady_clock::now();
    for (int test = 0; test < 50; test++) {
        do_test();
        cout << test << endl;
    }

    cout << "std_time: " << std_time << '\n';
    cout << "my_time: " << my_time << '\n';
    cout << "build_time: " << build_time << '\n';
    cout << "total_time: "
         << std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start
            )
                    .count() /
                1000.0;
}

int main() {
    // ifstream cin("input.txt");
    ios::sync_with_stdio(false), cin.tie(0), cout.tie(0);

    testing();
    return 0;

    uint32_t t = 33 * 1024;

    vector<uint32_t> a = {t, 5, t, 3};
    vector<uint32_t> b = {123, 5, t, t};

    cout << "correct: "
         << inner_product(
                a.begin(), a.end(), b.begin(), static_cast<uint32_t>(0)
            )
         << '\n';

    const __m128i zero128 = _mm_setzero_si128();

    __m128i a128 = _mm_load_si128(reinterpret_cast<__m128i *>(a.data()));
    __m128i b128 = _mm_load_si128(reinterpret_cast<__m128i *>(b.data()));

    {
        cout << '\n';
        uint32_t *aptr = reinterpret_cast<uint32_t *>(&a128);
        cout << aptr[0] << ' ' << aptr[1] << ' ' << aptr[2] << ' ' << aptr[3]
             << '\n';

        uint32_t *bptr = reinterpret_cast<uint32_t *>(&b128);
        cout << bptr[0] << ' ' << bptr[1] << ' ' << bptr[2] << ' ' << bptr[3]
             << '\n';

        cout << aptr[0] * bptr[0] << ' ' << aptr[1] * bptr[1] << ' '
             << aptr[2] * bptr[2] << ' ' << aptr[3] * bptr[3] << '\n';
    }

    {
        cout << '\n';
        __m128i kek = multiply128(a128, b128);
        auto t = reinterpret_cast<uint32_t *>(&kek);
        cout << t[0] << ' ' << t[1] << ' ' << t[2] << ' ' << t[3] << '\n';
    }
    return 0;

    __m128i a_lo32 = _mm_unpacklo_epi32(a128, zero128);
    __m128i a_hi32 = _mm_unpackhi_epi32(a128, zero128);

    __m128i b_lo32 = _mm_unpacklo_epi32(b128, zero128);
    __m128i b_hi32 = _mm_unpackhi_epi32(b128, zero128);

    {
        cout << '\n';
        auto t = reinterpret_cast<uint32_t *>(&a_lo32);
        cout << t[0] << ' ' << t[1] << ' ' << t[2] << ' ' << t[3] << '\n';

        auto p = reinterpret_cast<uint32_t *>(&a_hi32);
        cout << p[0] << ' ' << p[1] << ' ' << p[2] << ' ' << p[3] << '\n';
    }

    __m128i mul_lo32 = _mm_mul_epu32(a_lo32, b_lo32);
    __m128i mul_hi32 = _mm_mul_epu32(a_hi32, b_hi32);

    {
        cout << '\n';
        auto t = reinterpret_cast<uint32_t *>(&mul_lo32);
        cout << t[0] << ' ' << t[1] << ' ' << t[2] << ' ' << t[3] << '\n';

        auto p = reinterpret_cast<uint32_t *>(&mul_hi32);
        cout << p[0] << ' ' << p[1] << ' ' << p[2] << ' ' << p[3] << '\n';
    }

    __m128i sum = _mm_add_epi32(mul_lo32, mul_hi32);
    {
        uint32_t ans = 0;
        cout << '\n';
        auto t = reinterpret_cast<uint32_t *>(&sum);
        cout << t[0] << ' ' << t[1] << ' ' << t[2] << ' ' << t[3] << '\n';

        ans = t[0] + t[2];
        cout << "my_ans: " << ans << '\n';
    }

    //__m128i d = _mm_mul_epu32(a128, b128);
    // auto dptr = reinterpret_cast<uint32_t*>(&d);
    // cout << dptr[0] << ' ' << dptr[1] << ' ' << dptr[2] << ' ' << dptr[3] <<
    // '\n';
}
