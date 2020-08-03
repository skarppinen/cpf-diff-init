## Math-related functions.

using Statistics
using LinearAlgebra

logit(x::Real) = log(x / (1.0 - x));
invlogit(x::Real) = 1.0 / (1.0 + exp(-x));

"""
Expected square jump distance.
"""
function esjd(x::AVec{<: AVec{<: Real}})
    esjd = zero(eltype(eltype(x)));
    for i in 2:length(x)
        j = i - 1; # Index of distance computed at this iteration.
        d = L2dist(x[i], x[i - 1]); d2 = d * d;
        esjd = (j - 1) * esjd / j + d2 / j;
    end
    esjd;
end

function esjd(x::AVec{<: Real})
    esjd = zero(eltype(x)); # Current mean.
    for i in 2:length(x)
        j = i - 1; # Index of distance computed at this iteration.
        d2 = (x[i] - x[i - 1]) * (x[i] - x[i - 1]);
        esjd = (j - 1) * esjd / j + d2 / j;
    end
    esjd;
end

"""
    iact(x)

Calculate integrated autocorrelation of the sequence 'x' using an adaptive window
truncated autocorrelation sum estimator.
"""
function iact(x::AVec{<: Real})
    n = length(x);

    # Calculate standardised X.
    x_ = (x .- mean(x)) / sqrt(var(x));

    # The value C is suggested by Sokal according to
    # http://dfm.io/posts/autocorr/
    C = max(5.0, log10(n));

    # Compute the IACT by summing the autocorrelations
    # up to an index dependent on C.
    tau = 1.0;
    for k in 1:(n-1)
        tau += 2.0 * acf_standardised_x(x_, k);
        if k > C * tau
            break;
        end
    end
    tau;
end

"""
Compute the autocorrelation at lag `lag` for a univariate series `x`.
The series `x` is assumed standardised, eg. the mean has been subtracted
and the values have been divided by the standard deviation.
"""
function acf_standardised_x(x::AVec{<: Real}, lag::Int)
      n = length(x);
      lag < n || return 0.0
      dot(x, 1:(n - lag), x, (1 + lag):n) / (n - lag);
end

"""
Starting from the beginning of the vector `x`, compute the proportion (w.r.t
to the length of the vector - 1) of how many times the consecutive elements in the
vector were not equal.

Examples:
`prop_of_changes([1, 1, 1])` == 0.0 (no changes)
`prop_of_changes([1, 2, 3])` == 1.0 (change at every consecutive element)
`prop_of_changes([1, 1, 2])` == 0.5 (one change of a maximum of two)
`prop_of_changes([1, 1, 2, 2])` == 0.333.. (one change of a maximum of three)
"""
function prop_of_changes(x::AVec)
   length(x) <= 1 && (return 0.0);
   n_changes = reduce(x, init = (x[1], 0)) do x, y
      x[1] == y && (return x;)
      (y, x[2] + 1);
   end[2];
   n_changes / (length(x) - 1);
end
