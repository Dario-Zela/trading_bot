# Extend the official Metabase image with libstdc++.
#
# The DuckDB JNI driver (community-maintained, motherduckdb/metabase_duckdb_driver)
# ships a native libduckdb_java.so that was built against glibc/libstdc++.
# Metabase's official image is Alpine-based (musl libc), which does NOT
# include libstdc++ by default. Loading the driver fails at connection time:
#
#   Error loading shared library libstdc++.so.6: No such file or directory
#     (needed by /tmp/libduckdb_java*.so)
#
# `apk add libstdc++` provides the compat layer the JNI blob needs.

FROM metabase/metabase:latest

USER root
RUN apk add --no-cache libstdc++
USER metabase
