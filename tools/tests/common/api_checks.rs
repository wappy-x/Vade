use super::*;
#[path = "api_server.rs"] mod server;
#[test] fn rust_client_reads_chunked_streams_and_sends_tool_and_cache_turns() {
    let s = server::Server::start(server::replies());
    let mut models = String::new(); call(&s.url,"/v1/models",None).unwrap().read_to_string(&mut models).unwrap();
    assert_eq!(parse(&models).unwrap().get("data").get("0").get("id").text().unwrap(),"test-model");
    stream(&s.url,"test-model").unwrap(); tools(&s.url,"test-model").unwrap(); cache(&s.url,"test-model").unwrap();
    server::check(&s.finish());
}
#[test] fn http_body_rejects_a_short_content_length_and_an_invalid_chunk() {
    for response in ["HTTP/1.1 200 OK\r\nContent-Length: 9\r\n\r\nshort", "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nz\r\n"] {
        let s = server::Server::raw(vec![response.as_bytes().to_vec()]);
        let mut text = String::new(); let result = call(&s.url,"/v1/models",None).unwrap().read_to_string(&mut text);
        s.finish(); assert!(result.is_err());
    }
}
