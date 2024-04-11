# functions for downloading files

function download_file {
  local url="$1"
  local out_path="$2"
  local decompress="${3:-false}"

  if [ -f "$out_path" ]; then
    echo "$out_path exists, skipping" >&2
    return 17 # file exists
  else
    echo "Downloading $out_path" >&2
    # make parent directory if it doesn't exist
    local parent_dir=$(dirname "$out_path")
    if [ ! -d "$parent_dir" ]; then
      mkdir --parents "$parent_dir"
    fi

    mkfifo dl
    curl "$url" > dl &

    if [ "$decompress" == "true" ]; then
      gzip --stdout --decompress < dl > "$out_path"
    else
      cat < dl > "$out_path"
    fi

    rm dl
    # make data read-only
    chmod -w "$out_path"
  fi
}


function jgi_login() {
  local cookies="${1:-jgi_cookies.txt}"
  local jgi_user
  local jgi_password

  if [ ! -f "$cookies" ] || (! grep $'\tjgi_session\t' "$cookies" > /dev/null); then
    read -p "JGI Phytozome Username: " jgi_user
    read -s -p "JGI Phytozome Password: " jgi_password
    echo

    curl 'https://signon.jgi.doe.gov/signon/create' --data-urlencode "login=${jgi_user}" --data-urlencode "password=${jgi_password}" --cookie-jar "$cookies" > /dev/null 2>&1

    if (grep $'\tjgi_session\t' "$cookies" > /dev/null); then
      return 0
    else
      echo "Failed to log in to JGI, incorrect password?" >&2
      rm "$cookies"
      return 1
    fi
  fi
}


function jgi_download() {
  local file_id="$1"
  local out_path="$2"
  local decompress="${3:-false}"
  local cookies="${4:-jgi_cookies.txt}"

  if [ -f "$out_path" ]; then
    echo "$out_path already exists, skipping" >&2
    return 17 # file exists
  fi

  if ! jgi_login; then
    return 1
  fi

  # make parent directory if it doesn't exist
  local parent_dir=$(dirname "$out_path")
  if [ ! -d "$parent_dir" ]; then
    mkdir --parents "$parent_dir"
  fi

  echo "Downloading $file_id to $out_path" >&2
  curl \
    --fail \
    --cookie "$cookies" \
    --header 'accept: application/json' \
    "https://files.jgi.doe.gov/download_files/${file_id}/" > "$out_path"

  if [ "$decompress" == "true" ]; then
    gzip --decompress "$out_path"
  fi

  # make data read-only
  chmod -w "$out_path"

  # rate-limit
  sleep 10

  return $?
}


function jgi_logout() {
  local cookies="${1:-jgi_cookies.txt}"
  rm --force "$cookies"
}


function cyverse_download() {
  local path="$1"
  local out_path="$2"

  if [ -f "$out_path" ]; then
    echo "$out_path exists, skipping" >&2
    return 17 # file exists
  else
    echo "Downloading $out_path" >&2
    python -c 'import sys, pathlib, irods.session; path = pathlib.Path(sys.argv[1]); session = irods.session.iRODSSession(host = "data.cyverse.org", port = 1247, user = "anonymous", password = "", zone = path.parts[1]); session.data_objects.get(str(path), sys.argv[2] if ((len(sys.argv) > 2) and sys.argv[2]) else path.name); session.cleanup()' "$path" "$out_path"
  fi

  return $?
}
