## Multinomial resampling.
# The implementation is based on some code by Matti Vihola.

include("../../../config.jl");

"""
An abstract type representing a resampling algorithm.
"""
abstract type Resampling end
struct MultinomialResampling <: Resampling end

function resample!(ind::AbstractVector{Int64}, p::AbstractVector{<: Real}, mr::MultinomialResampling;
                   ref_prev::Int = 0, ref_cur::Int = 0)::Nothing
    m = length(p);
    n = length(ind);
    K = 1; @inbounds S = p[1];
    U_ = 1.0; n_ = n;
    for j in 1:n
      # Order statistics in reverse order:
      U_ = rand() ^ (1.0 / n_) * U_;
      n_ -= 1;
      # ...same as forward order by symmetry...
      U = 1.0 - U_;
      # Find K such that F(K) >= u
      while K < m && U > S
        K = K + 1;       # Note that K is not reset!
        @inbounds S = S + p[K]; # S is the partial sum up to K
      end
      @inbounds ind[j] = K;
    end

    if ref_prev > 0
      # Need swap or shuffle because of sampling uniforms in order.
      #swap!(ind, ref_cur, sample(Base.OneTo(m)));
      shuffle!(ind);
      @inbounds ind[ref_cur] = ref_prev;
    end
    nothing
end

function swap!(x::AbstractVector, i::Integer, j::Integer)
  @inbounds tmp = x[i];
  @inbounds x[i] = x[j];
  @inbounds x[j] = tmp;
  nothing
end
