module Statespace
export SSM, SSMInstance, SSMStorage

abstract type SSMStorage end
abstract type SSM end

"""
A function each type of state space model must import and define.
The function should return the type name of the correct storage type
for the model.
"""
function storagetype()

end

"""
An object putting together:
1. A state space model definition (functions that define the model, no data).
2. A storage object that is used when making model related computations.
   Preallocated to remove unwanted allocations.
   The storage object must be compatible with the type of the state space model.
   This is checked.
3. Any kind of data associated with the state space model.
4. The length of the time series. Depending on the model, this is not always
   defined by the `data` or the dimensions in `storage`.
   
NOTE: No data copying is made as an SSMInstance is constructed.
It is possible to have multiple state space models referencing the same data and
storage space to save memory.
It is also possible to run multiple models on the same data/storage without
making copies.
"""
struct SSMInstance{M <: SSM, S <: SSMStorage, D <: Any}
  model::M
  storage::S
  data::D
  len::Int
  function SSMInstance(m::SSM, s::SSMStorage, d, len::Integer)
    @assert typeof(s) <: storagetype(m)
    @assert len > 0 "the length of the timeseries must be > 0.";
    new{typeof(m), typeof(s), typeof(d)}(m, s, d, len);
  end
end

import Base.length
function length(ssm::SSMInstance)
  ssm.len;
end

end
