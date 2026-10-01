_MAX_ATTEMPTS = 3

def _archive_name(url):
    return url.split("?")[0].split("/")[-1]

def _download_ranges(ctx, url, ranges):
    """Downloads (part, lo, hi) ranges concurrently, retrying whole parts that fail.

    Bazel retries an interrupted download by resuming it, which sends its own Range header. That
    both collides with ours and asks for the wrong offset, since it measures resumption from the
    size of the part file rather than from the start of our range. So we let a part fail instead
    (allow_fail) and restart it from scratch here.
    """
    todo = ranges
    for _ in range(_MAX_ATTEMPTS):
        pending = []
        for part, lo, hi in todo:
            ctx.delete(part)
            pending.append((part, lo, hi, ctx.download(
                url = url,
                output = part,
                block = False,
                allow_fail = True,
                headers = {"Range": "bytes={}-{}".format(lo, hi)},
            )))

        failed = []
        for part, lo, hi, download in pending:
            if not download.wait().success:
                failed.append((part, lo, hi))

        if not failed:
            return
        todo = failed
        ctx.report_progress("Retrying {} failed range(s) of {}".format(len(failed), url))

    fail("Failed to download {} after {} attempts".format(url, _MAX_ATTEMPTS))

def _parallel_http_archive_impl(ctx):
    archive = _archive_name(ctx.attr.url)
    size = int(ctx.attr.size_bytes)
    chunk = ctx.attr.chunk_bytes

    # Bazel downloads with one connection per URL, which the HuggingFace CDN caps at ~14 MB/s.
    # Ranged downloads in parallel reach ~105 MB/s. The Range header survives HF's redirect to
    # its signed CDN URL.
    #
    # Chunks are small and numerous rather than one per connection: the CDN throttles individual
    # connections unpredictably, and with one long chunk per connection a single slow one sets the
    # wall time while the rest sit idle (observed at 0.2 MB/s against 100 MB/s available). Queueing
    # many short chunks through bazel's download pool keeps every slot busy and caps the damage from
    # a slow chunk at one chunk.
    count = (size + chunk - 1) // chunk
    ctx.report_progress("Downloading {} ({} bytes) in {} chunks".format(archive, size, count))
    ranges = []
    for i in range(count):
        lo = i * chunk
        ranges.append(("{}.part{}".format(archive, i), lo, min(lo + chunk, size) - 1))

    _download_ranges(ctx, ctx.attr.url, ranges)
    parts = [part for part, _, _ in ranges]

    ctx.report_progress("Assembling and verifying {}".format(archive))
    concat = ctx.execute(["bash", "-c", "cat {} > {}".format(" ".join(parts), archive)])
    if concat.return_code != 0:
        fail("Failed to concatenate parts of {}: {}".format(archive, concat.stderr))

    # The ranges are unverified individually, so the pin is enforced here. sha256sum emits hex and
    # the pin is base64, hence the xxd round trip.
    digest = ctx.execute(["bash", "-c", "sha256sum {} | cut -d' ' -f1 | xxd -r -p | base64 -w0".format(archive)])
    if digest.return_code != 0:
        fail("Failed to hash {}: {}".format(archive, digest.stderr))

    actual = "sha256-" + digest.stdout.strip()
    if actual != ctx.attr.integrity:
        fail("Checksum mismatch for {}\n  expected: {}\n  actual:   {}".format(
            ctx.attr.url,
            ctx.attr.integrity,
            actual,
        ))

    for part in parts:
        ctx.delete(part)

    ctx.report_progress("Extracting {}".format(archive))
    ctx.extract(archive = archive, stripPrefix = ctx.attr.strip_prefix)
    ctx.delete(archive)
    ctx.file("BUILD.bazel", ctx.attr.build_file_content)

parallel_http_archive = repository_rule(
    implementation = _parallel_http_archive_impl,
    doc = """Fetches and extracts an archive using concurrent HTTP range requests.

A drop-in replacement for `http_archive` when one connection, not the link, is the bottleneck.
`size_bytes` is extra because the ranges must be computed before the first request; `integrity` is
mandatory and checked against the assembled file, so a wrong `size_bytes` fails the checksum
instead of yielding a truncated archive.

This rule does not populate the content-addressable repository cache. Reuse comes from the repo
contents cache instead.
""",
    attrs = {
        "url": attr.string(mandatory = True, doc = "URL of the archive to fetch"),
        "integrity": attr.string(mandatory = True, doc = "Expected checksum in Subresource Integrity format"),
        # Not named `size`: that collides with bazel's native (configurable) test-size attribute,
        # which a repo rule rejects outright. A string rather than an int because attr.int is
        # limited to the signed 32-bit range and these archives are well past 2 GB.
        "size_bytes": attr.string(mandatory = True, doc = "Total size of the archive in bytes"),
        "chunk_bytes": attr.int(default = 64 * 1024 * 1024, doc = "Size of each ranged request"),
        "strip_prefix": attr.string(doc = "Directory prefix to strip from the extracted files"),
        "build_file_content": attr.string(doc = "Content of the BUILD file generated for the repo"),
    },
)
