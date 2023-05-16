// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#define R_NO_REMAP
#include <R.h>
#include <Rinternals.h>

#include "altrep.h"
#include "util.h"

/* generated by tools/make-callentries.R */
extern SEXP nanoarrow_c_make_altrep_chr(SEXP array_xptr);
extern SEXP nanoarrow_c_is_altrep(SEXP x_sexp);
extern SEXP nanoarrow_c_altrep_is_materialized(SEXP x_sexp);
extern SEXP nanoarrow_c_altrep_force_materialize(SEXP x_sexp, SEXP recursive_sexp);
extern SEXP nanoarrow_c_array_stream_get_schema(SEXP array_stream_xptr);
extern SEXP nanoarrow_c_array_stream_get_next(SEXP array_stream_xptr);
extern SEXP nanoarrow_c_basic_array_stream(SEXP batches_sexp, SEXP schema_xptr, SEXP validate_sexp);
extern SEXP nanoarrow_c_array_view(SEXP array_xptr, SEXP schema_xptr);
extern SEXP nanoarrow_c_array_init(SEXP schema_xptr);
extern SEXP nanoarrow_c_array_set_length(SEXP array_xptr, SEXP length_sexp);
extern SEXP nanoarrow_c_array_set_null_count(SEXP array_xptr, SEXP null_count_sexp);
extern SEXP nanoarrow_c_array_set_offset(SEXP array_xptr, SEXP offset_sexp);
extern SEXP nanoarrow_c_array_set_buffers(SEXP array_xptr, SEXP buffers_sexp);
extern SEXP nanoarrow_c_array_set_children(SEXP array_xptr, SEXP children_sexp);
extern SEXP nanoarrow_c_array_set_dictionary(SEXP array_xptr, SEXP dictionary_xptr);
extern SEXP nanoarrow_c_array_validate_after_modify(SEXP array_xptr, SEXP schema_xptr);
extern SEXP nanoarrow_c_array_set_schema(SEXP array_xptr, SEXP schema_xptr, SEXP validate_sexp);
extern SEXP nanoarrow_c_infer_schema_array(SEXP array_xptr);
extern SEXP nanoarrow_c_array_proxy(SEXP array_xptr, SEXP array_view_xptr, SEXP recursive_sexp);
extern SEXP nanoarrow_c_as_array_default(SEXP x_sexp, SEXP schema_sexp);
extern SEXP nanoarrow_c_as_buffer_default(SEXP x_sexp);
extern SEXP nanoarrow_c_buffer_append(SEXP buffer_xptr, SEXP new_buffer_xptr);
extern SEXP nanoarrow_c_buffer_info(SEXP buffer_xptr);
extern SEXP nanoarrow_c_buffer_as_raw(SEXP buffer_xptr);
extern SEXP nanoarrow_c_convert_array_stream(SEXP array_stream_xptr, SEXP ptype_sexp, SEXP size_sexp, SEXP n_sexp);
extern SEXP nanoarrow_c_convert_array(SEXP array_xptr, SEXP ptype_sexp);
extern SEXP nanoarrow_c_infer_ptype(SEXP schema_xptr);
extern SEXP nanoarrow_c_allocate_schema(void);
extern SEXP nanoarrow_c_allocate_array(void);
extern SEXP nanoarrow_c_allocate_array_stream(void);
extern SEXP nanoarrow_c_pointer(SEXP obj_sexp);
extern SEXP nanoarrow_c_pointer_addr_dbl(SEXP ptr);
extern SEXP nanoarrow_c_pointer_addr_chr(SEXP ptr);
extern SEXP nanoarrow_c_pointer_addr_pretty(SEXP ptr);
extern SEXP nanoarrow_c_pointer_is_valid(SEXP ptr);
extern SEXP nanoarrow_c_pointer_release(SEXP ptr);
extern SEXP nanoarrow_c_pointer_move(SEXP ptr_src, SEXP ptr_dst);
extern SEXP nanoarrow_c_export_schema(SEXP schema_xptr, SEXP ptr_dst);
extern SEXP nanoarrow_c_export_array(SEXP array_xptr, SEXP ptr_dst);
extern SEXP nanoarrow_c_export_array_stream(SEXP array_stream_xptr, SEXP ptr_dst);
extern SEXP nanoarrow_c_pointer_set_protected(SEXP ptr_src, SEXP protected_sexp);
extern SEXP nanoarrow_c_schema_init(SEXP type_id_sexp, SEXP nullable_sexp);
extern SEXP nanoarrow_c_schema_init_date_time(SEXP type_id_sexp, SEXP time_unit_sexp, SEXP timezone_sexp, SEXP nullable_sexp);
extern SEXP nanoarrow_c_schema_init_decimal(SEXP type_id_sexp, SEXP precision_sexp, SEXP scale_sexp, SEXP nullable_sexp);
extern SEXP nanoarrow_c_schema_init_fixed_size(SEXP type_id_sexp, SEXP fixed_size_sexp, SEXP nullable_sexp);
extern SEXP nanoarrow_c_schema_to_list(SEXP schema_xptr);
extern SEXP nanoarrow_c_schema_parse(SEXP schema_xptr);
extern SEXP nanoarrow_c_schema_format(SEXP schema_xptr, SEXP recursive_sexp);
extern SEXP nanoarrow_c_schema_set_format(SEXP schema_mut_xptr, SEXP format_sexp);
extern SEXP nanoarrow_c_schema_set_name(SEXP schema_mut_xptr, SEXP name_sexp);
extern SEXP nanoarrow_c_schema_set_metadata(SEXP schema_mut_xptr, SEXP metadata_sexp);
extern SEXP nanoarrow_c_schema_set_flags(SEXP schema_mut_xptr, SEXP flags_sexp);
extern SEXP nanoarrow_c_schema_set_children(SEXP schema_mut_xptr, SEXP children_sexp);
extern SEXP nanoarrow_c_schema_set_dictionary(SEXP schema_mut_xptr, SEXP dictionary_xptr);
extern SEXP nanoarrow_c_preserved_count(void);
extern SEXP nanoarrow_c_preserved_empty(void);
extern SEXP nanoarrow_c_preserve_and_release_on_other_thread(SEXP obj);
extern SEXP nanoarrow_c_version(void);
extern SEXP nanoarrow_c_version_runtime(void);

static const R_CallMethodDef CallEntries[] = {
    {"nanoarrow_c_make_altrep_chr", (DL_FUNC)&nanoarrow_c_make_altrep_chr, 1},
    {"nanoarrow_c_is_altrep", (DL_FUNC)&nanoarrow_c_is_altrep, 1},
    {"nanoarrow_c_altrep_is_materialized", (DL_FUNC)&nanoarrow_c_altrep_is_materialized, 1},
    {"nanoarrow_c_altrep_force_materialize", (DL_FUNC)&nanoarrow_c_altrep_force_materialize, 2},
    {"nanoarrow_c_array_stream_get_schema", (DL_FUNC)&nanoarrow_c_array_stream_get_schema, 1},
    {"nanoarrow_c_array_stream_get_next", (DL_FUNC)&nanoarrow_c_array_stream_get_next, 1},
    {"nanoarrow_c_basic_array_stream", (DL_FUNC)&nanoarrow_c_basic_array_stream, 3},
    {"nanoarrow_c_array_view", (DL_FUNC)&nanoarrow_c_array_view, 2},
    {"nanoarrow_c_array_init", (DL_FUNC)&nanoarrow_c_array_init, 1},
    {"nanoarrow_c_array_set_length", (DL_FUNC)&nanoarrow_c_array_set_length, 2},
    {"nanoarrow_c_array_set_null_count", (DL_FUNC)&nanoarrow_c_array_set_null_count, 2},
    {"nanoarrow_c_array_set_offset", (DL_FUNC)&nanoarrow_c_array_set_offset, 2},
    {"nanoarrow_c_array_set_buffers", (DL_FUNC)&nanoarrow_c_array_set_buffers, 2},
    {"nanoarrow_c_array_set_children", (DL_FUNC)&nanoarrow_c_array_set_children, 2},
    {"nanoarrow_c_array_set_dictionary", (DL_FUNC)&nanoarrow_c_array_set_dictionary, 2},
    {"nanoarrow_c_array_validate_after_modify", (DL_FUNC)&nanoarrow_c_array_validate_after_modify, 2},
    {"nanoarrow_c_array_set_schema", (DL_FUNC)&nanoarrow_c_array_set_schema, 3},
    {"nanoarrow_c_infer_schema_array", (DL_FUNC)&nanoarrow_c_infer_schema_array, 1},
    {"nanoarrow_c_array_proxy", (DL_FUNC)&nanoarrow_c_array_proxy, 3},
    {"nanoarrow_c_as_array_default", (DL_FUNC)&nanoarrow_c_as_array_default, 2},
    {"nanoarrow_c_as_buffer_default", (DL_FUNC)&nanoarrow_c_as_buffer_default, 1},
    {"nanoarrow_c_buffer_append", (DL_FUNC)&nanoarrow_c_buffer_append, 2},
    {"nanoarrow_c_buffer_info", (DL_FUNC)&nanoarrow_c_buffer_info, 1},
    {"nanoarrow_c_buffer_as_raw", (DL_FUNC)&nanoarrow_c_buffer_as_raw, 1},
    {"nanoarrow_c_convert_array_stream", (DL_FUNC)&nanoarrow_c_convert_array_stream, 4},
    {"nanoarrow_c_convert_array", (DL_FUNC)&nanoarrow_c_convert_array, 2},
    {"nanoarrow_c_infer_ptype", (DL_FUNC)&nanoarrow_c_infer_ptype, 1},
    {"nanoarrow_c_allocate_schema", (DL_FUNC)&nanoarrow_c_allocate_schema, 0},
    {"nanoarrow_c_allocate_array", (DL_FUNC)&nanoarrow_c_allocate_array, 0},
    {"nanoarrow_c_allocate_array_stream", (DL_FUNC)&nanoarrow_c_allocate_array_stream, 0},
    {"nanoarrow_c_pointer", (DL_FUNC)&nanoarrow_c_pointer, 1},
    {"nanoarrow_c_pointer_addr_dbl", (DL_FUNC)&nanoarrow_c_pointer_addr_dbl, 1},
    {"nanoarrow_c_pointer_addr_chr", (DL_FUNC)&nanoarrow_c_pointer_addr_chr, 1},
    {"nanoarrow_c_pointer_addr_pretty", (DL_FUNC)&nanoarrow_c_pointer_addr_pretty, 1},
    {"nanoarrow_c_pointer_is_valid", (DL_FUNC)&nanoarrow_c_pointer_is_valid, 1},
    {"nanoarrow_c_pointer_release", (DL_FUNC)&nanoarrow_c_pointer_release, 1},
    {"nanoarrow_c_pointer_move", (DL_FUNC)&nanoarrow_c_pointer_move, 2},
    {"nanoarrow_c_export_schema", (DL_FUNC)&nanoarrow_c_export_schema, 2},
    {"nanoarrow_c_export_array", (DL_FUNC)&nanoarrow_c_export_array, 2},
    {"nanoarrow_c_export_array_stream", (DL_FUNC)&nanoarrow_c_export_array_stream, 2},
    {"nanoarrow_c_pointer_set_protected", (DL_FUNC)&nanoarrow_c_pointer_set_protected, 2},
    {"nanoarrow_c_schema_init", (DL_FUNC)&nanoarrow_c_schema_init, 2},
    {"nanoarrow_c_schema_init_date_time", (DL_FUNC)&nanoarrow_c_schema_init_date_time, 4},
    {"nanoarrow_c_schema_init_decimal", (DL_FUNC)&nanoarrow_c_schema_init_decimal, 4},
    {"nanoarrow_c_schema_init_fixed_size", (DL_FUNC)&nanoarrow_c_schema_init_fixed_size, 3},
    {"nanoarrow_c_schema_to_list", (DL_FUNC)&nanoarrow_c_schema_to_list, 1},
    {"nanoarrow_c_schema_parse", (DL_FUNC)&nanoarrow_c_schema_parse, 1},
    {"nanoarrow_c_schema_format", (DL_FUNC)&nanoarrow_c_schema_format, 2},
    {"nanoarrow_c_schema_set_format", (DL_FUNC)&nanoarrow_c_schema_set_format, 2},
    {"nanoarrow_c_schema_set_name", (DL_FUNC)&nanoarrow_c_schema_set_name, 2},
    {"nanoarrow_c_schema_set_metadata", (DL_FUNC)&nanoarrow_c_schema_set_metadata, 2},
    {"nanoarrow_c_schema_set_flags", (DL_FUNC)&nanoarrow_c_schema_set_flags, 2},
    {"nanoarrow_c_schema_set_children", (DL_FUNC)&nanoarrow_c_schema_set_children, 2},
    {"nanoarrow_c_schema_set_dictionary", (DL_FUNC)&nanoarrow_c_schema_set_dictionary, 2},
    {"nanoarrow_c_preserved_count", (DL_FUNC)&nanoarrow_c_preserved_count, 0},
    {"nanoarrow_c_preserved_empty", (DL_FUNC)&nanoarrow_c_preserved_empty, 0},
    {"nanoarrow_c_preserve_and_release_on_other_thread", (DL_FUNC)&nanoarrow_c_preserve_and_release_on_other_thread, 1},
    {"nanoarrow_c_version", (DL_FUNC)&nanoarrow_c_version, 0},
    {"nanoarrow_c_version_runtime", (DL_FUNC)&nanoarrow_c_version_runtime, 0},
    {NULL, NULL, 0}};
/* end generated by tools/make-callentries.R */

void R_init_nanoarrow(DllInfo* dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);

  nanoarrow_init_cached_sexps();
  nanoarrow_preserve_init();
  register_nanoarrow_altrep(dll);
}
