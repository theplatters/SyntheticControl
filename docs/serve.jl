using Sockets

root = normpath(joinpath(@__DIR__, "build"))
isdir(root) || error("docs/build does not exist; run `julia --project=docs docs/make.jl` first")

port = parse(Int, get(ENV, "DOCS_PORT", "8000"))
server = Sockets.listen(Sockets.localhost, port)
println("Serving documentation from $root")
println("Open http://127.0.0.1:$port/")
println("Press Ctrl-C to stop.")

const MIME_TYPES = Dict(
  ".css" => "text/css; charset=utf-8",
  ".html" => "text/html; charset=utf-8",
  ".js" => "application/javascript; charset=utf-8",
  ".json" => "application/json; charset=utf-8",
  ".svg" => "image/svg+xml",
  ".png" => "image/png",
  ".ico" => "image/x-icon",
  ".inv" => "application/octet-stream",
)

function response(client, status, content_type, body)
  write(client, "HTTP/1.1 $status\r\n")
  write(client, "Content-Type: $content_type\r\n")
  write(client, "Content-Length: $(sizeof(body))\r\n")
  write(client, "Connection: close\r\n\r\n")
  write(client, body)
end

function safe_path(root, target)
  path = split(target, '?'; limit=2)[1]
  path = isempty(path) || path == "/" ? "/index.html" : path
  path = replace(path, "%20" => " ")
  candidate = normpath(joinpath(root, lstrip(path, '/')))
  startswith(candidate, root) || return nothing
  isdir(candidate) && (candidate = joinpath(candidate, "index.html"))
  return candidate
end

try
  while true
    client = accept(server)
    errormonitor(@async begin
      try
        request_line = readline(client)
        isempty(request_line) && return
        parts = split(request_line)
        length(parts) >= 2 || return
        target = parts[2]
        while !isempty(readline(client))
        end

        path = safe_path(root, target)
        if path === nothing || !isfile(path)
          response(client, "404 Not Found", "text/plain; charset=utf-8", "Not found")
        else
          ext = lowercase(splitext(path)[2])
          content_type = get(MIME_TYPES, ext, "application/octet-stream")
          response(client, "200 OK", content_type, read(path))
        end
      finally
        close(client)
      end
    end)
  end
catch err
  err isa InterruptException || rethrow()
  println("\nStopped documentation server.")
finally
  close(server)
end
