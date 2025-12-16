pub const ApiError = error{
    MethodNotAllowed,
    InvalidJson,
    MissingField,
    MissingBody,
    InternalFailure,
};
