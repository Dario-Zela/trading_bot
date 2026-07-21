# Metabase on a Debian JRE base, not Alpine.
#
# Why not extend metabase/metabase:latest directly:
#   The official image is Alpine (musl libc). The community DuckDB
#   Metabase driver ships libduckdb_java.so linked against glibc.
#   Progressive failure modes we already hit trying to make Alpine work:
#     1. libstdc++.so.6 missing            → fixed with `apk add libstdc++`
#     2. Class-init failure (glibc symbols) → fixed with `apk add gcompat`
#     3. JVM segfault on actual connection → gcompat can't bridge every
#                                            glibc symbol the JNI blob
#                                            calls, so runtime calls
#                                            still crash the JVM
#
# The durable fix is a glibc host. This Dockerfile takes the official
# Metabase jar (from their Alpine image) and runs it on Debian —
# the jar itself is JVM bytecode and doesn't care about the OS, only
# the native libraries the JNI drivers load do. On Debian all
# glibc-linked native libraries work directly.

# Pinned to v0.55.12: MotherDuck's DuckDB driver 1.5.x is built against
# Metabase v0.55's plugin API. Newer Metabase (v0.63+) sends connection
# options the driver doesn't know (`connection-pool-type` etc.) and
# every attempt to open a DuckDB connection fails with:
#   Invalid Input Error: The following options were not recognized: ...
# Pinning both sides to the pair they were tested against sidesteps
# the mismatch. Bump both together when MotherDuck releases a
# Metabase-v0.63-targeting driver.
FROM metabase/metabase:v0.55.12 AS source

FROM eclipse-temurin:21-jre-noble

# curl needed for the docker-compose healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# Match the container UID/GID of the upstream image so bind-mounted
# volumes chown'd to 2000 in post-start.sh are writable.
RUN groupadd -g 2000 metabase \
 && useradd  -u 2000 -g metabase -m -s /bin/bash metabase

# Copy Metabase's runtime layout from the official image. The jar is
# at /app/metabase.jar, plus supporting scripts under /app.
COPY --from=source --chown=metabase:metabase /app /app

USER metabase
WORKDIR /app

EXPOSE 3000

# Same entrypoint the upstream image uses.
CMD ["java", "-jar", "/app/metabase.jar"]
