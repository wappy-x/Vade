use std::{io::{self,BufRead,BufReader,Write},net::TcpListener,sync::{Arc,atomic::{AtomicBool,Ordering}},thread,time::{Duration,Instant}};
pub struct Server { pub url: String, task: Option<thread::JoinHandle<io::Result<Vec<String>>>>, stop: Arc<AtomicBool> }
fn request(reader: &mut impl BufRead) -> io::Result<String> {
    let mut request = String::new(); let mut length = 0;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 { return Err(io::Error::new(io::ErrorKind::UnexpectedEof,"incomplete request headers")); }
        if line == "\r\n" { break; }
        if let Some((key,value)) = line.split_once(':') { if key.eq_ignore_ascii_case("content-length") { length = value.trim().parse().map_err(io::Error::other)?; } }
        request.push_str(&line);
    }
    let mut data = vec![0;length]; reader.read_exact(&mut data)?;
    request.push_str(&String::from_utf8(data).map_err(io::Error::other)?); Ok(request)
}
impl Server {
    pub fn start(replies: Vec<(String,bool)>) -> Self {
        Self::raw(replies.into_iter().map(|(body,chunked)| {
            if !chunked { return format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",body.len()).into_bytes(); }
            let mut response = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n".to_vec();
            for chunk in body.as_bytes().chunks(7) { response.extend_from_slice(format!("{:x};test=1\r\n",chunk.len()).as_bytes()); response.extend_from_slice(chunk); response.extend_from_slice(b"\r\n"); }
            response.extend_from_slice(b"0\r\n\r\n"); response
        }).collect())
    }
    pub fn raw(replies: Vec<Vec<u8>>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap(); listener.set_nonblocking(true).unwrap();
        let url = format!("http://{}",listener.local_addr().unwrap());
        let stop = Arc::new(AtomicBool::new(false)); let stopped = stop.clone();
        let task = thread::spawn(move || {
            let mut requests = Vec::new();
            for response in replies {
                let start = Instant::now();
                let mut stream = loop {
                    if stopped.load(Ordering::Relaxed) { return Err(io::Error::new(io::ErrorKind::Interrupted,"mock API stopped")); }
                    match listener.accept() { Ok((s,_)) => break s,
                    Err(e) if e.kind() == std::io::ErrorKind::WouldBlock && start.elapsed() < Duration::from_secs(10) => thread::sleep(Duration::from_millis(5)),
                    Err(e) => return Err(e) } };
                stream.set_nonblocking(false)?;
                stream.set_read_timeout(Some(Duration::from_secs(2)))?; stream.set_write_timeout(Some(Duration::from_secs(2)))?;
                requests.push(request(&mut BufReader::new(stream.try_clone()?))?);
                stream.write_all(&response)?;
            } Ok(requests)
        }); Self {url,task:Some(task),stop}
    }
    pub fn finish_result(mut self) -> io::Result<Vec<String>> { self.task.take().unwrap().join().unwrap() }
    pub fn finish(self) -> Vec<String> { self.finish_result().unwrap() }
}
impl Drop for Server { fn drop(&mut self) { self.stop.store(true,Ordering::Relaxed); if let Some(task) = self.task.take() { let _ = task.join(); } } }
#[test] fn mock_api_rejects_connect_and_close_and_incomplete_headers() {
    for prefix in ["","GET / HTTP/1.1\r\nHost: local\r\n"] {
        let s = Server::raw(vec![Vec::new()]);
        let mut tcp = std::net::TcpStream::connect(s.url.strip_prefix("http://").unwrap()).unwrap();
        tcp.write_all(prefix.as_bytes()).unwrap(); drop(tcp);
        assert_eq!(s.finish_result().unwrap_err().kind(),io::ErrorKind::UnexpectedEof);
    }
}
#[test] fn mock_api_drop_stops_and_joins_an_unused_server() { drop(Server::raw(vec![Vec::new()])); }
#[test] fn mock_api_waits_for_delayed_headers_and_body() {
    let s = Server::raw(vec![Vec::new()]);
    let mut tcp = std::net::TcpStream::connect(s.url.strip_prefix("http://").unwrap()).unwrap();
    tcp.set_write_timeout(Some(Duration::from_secs(2))).unwrap();
    for part in ["POST / HTTP/1.1\r\nHost: local\r\n", "Content-Length: 5\r\n\r\nhe", "llo"] {
        thread::sleep(Duration::from_millis(50)); tcp.write_all(part.as_bytes()).unwrap();
    }
    drop(tcp);
    assert_eq!(s.finish(),vec!["POST / HTTP/1.1\r\nHost: local\r\nContent-Length: 5\r\nhello"]);
}
pub fn replies() -> Vec<(String,bool)> {
    vec![(r#"{"data":[{"id":"test-model"}]}"#.into(),false),
        ("data: {\"choices\":[{\"delta\":{\"content\":\"hello é\\n\"}}]}\r\n\r\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2}}\n\ndata: [DONE]\n\n".into(),true),
        (r#"{"choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]}}]}"#.into(),true),
        (r#"{"choices":[{"message":{"role":"assistant","content":"Paris is clear."}}]}"#.into(),false),
        (r#"{"choices":[{"message":{"role":"assistant","content":"17"}}],"usage":{"prompt_tokens":900,"prompt_tokens_details":{"cached_tokens":0}}}"#.into(),false),
        (r#"{"choices":[{"message":{"role":"assistant","content":"19"}}],"usage":{"prompt_tokens":905,"prompt_tokens_details":{"cached_tokens":899}}}"#.into(),false)]
}
pub fn check(requests: &[String]) {
    assert_eq!(requests.len(),6); assert!(requests[0].starts_with("GET /v1/models "));
    let bodies: Vec<_> = requests[1..].iter().map(|r|super::json::parse(r.lines().last().unwrap()).unwrap()).collect();
    assert_eq!(bodies[0].get("stream"),&super::json::Value::Bool(true));
    let tool = bodies[2].get("messages").get("2"); assert_eq!(tool.get("tool_call_id").text().unwrap(),"call_1");
    assert_eq!(super::json::parse(tool.get("content").text().unwrap()).unwrap().get("city").text().unwrap(),"Paris");
    assert_eq!(bodies[4].get("messages").get("2").get("content").text().unwrap(),"17");
    assert!(bodies[3].get("messages").get("0").get("content").text().unwrap().contains("Rule 120: keep function 120 under 130 lines."));
    assert_eq!(bodies[3].get("prompt_cache_key").text().unwrap(),"examples-review");
}
