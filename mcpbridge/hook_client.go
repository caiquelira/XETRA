package mcpbridge

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"
)

// loopbackHTTPClient returns an *http.Client that never consults HTTP_PROXY /
// HTTPS_PROXY environment variables, preventing MITM recording proxies from
// intercepting loopback 127.0.0.1 bridge requests.
func loopbackHTTPClient(timeout time.Duration) *http.Client {
	return &http.Client{
		Timeout:   timeout,
		Transport: &http.Transport{Proxy: nil},
	}
}

// RunHookClient implements the `artex hook` subcommand: it reads a CLI hook
// JSON payload from stdin, forwards it to the local mcpbridge HTTP server
// (bypassing any HTTP_PROXY/HTTPS_PROXY), and writes the JSON decision to stdout.
func RunHookClient(ctx context.Context, args []string, stdin io.Reader, stdout io.Writer) error {
	fs := flag.NewFlagSet("hook", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	bridgeURL := fs.String("url", os.Getenv("XETRA_BRIDGE_URL"), "bridge base or hook URL")
	token := fs.String("token", os.Getenv("XETRA_BRIDGE_TOKEN"), "bridge run token")
	event := fs.String("event", os.Getenv("XETRA_HOOK_EVENT"), "hook event (PreToolUse, PostToolUse, Stop)")
	cli := fs.String("cli", os.Getenv("XETRA_HOOK_CLI"), "CLI dialect (claude or agy)")
	if err := fs.Parse(args); err != nil {
		return err
	}

	if strings.TrimSpace(*bridgeURL) == "" {
		return errors.New("missing bridge URL (--url or XETRA_BRIDGE_URL)")
	}
	targetURL, err := buildHookURL(*bridgeURL, *token, *event, *cli)
	if err != nil {
		return err
	}

	payload, err := io.ReadAll(io.LimitReader(stdin, maxBodyBytes))
	if err != nil {
		return fmt.Errorf("read stdin: %w", err)
	}
	if len(bytes.TrimSpace(payload)) == 0 {
		payload = []byte("{}")
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, targetURL, bytes.NewReader(payload))
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if t := strings.TrimSpace(*token); t != "" {
		req.Header.Set("X-Xetra-Bridge-Token", t)
	}
	if e := strings.TrimSpace(*event); e != "" {
		req.Header.Set("X-Xetra-Hook-Event", e)
	}
	if c := strings.TrimSpace(*cli); c != "" {
		req.Header.Set("X-Xetra-Hook-CLI", c)
	}

	// Generous timeout so operator target-gate / approval waits (up to 300s+) do not trip prematurely.
	client := loopbackHTTPClient(10 * time.Minute)
	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("post hook: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, maxBodyBytes))
	if err != nil {
		return fmt.Errorf("read response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("bridge returned HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(respBody)))
	}

	trimmed := bytes.TrimSpace(respBody)
	if len(trimmed) > 0 {
		if _, err := stdout.Write(append(trimmed, '\n')); err != nil {
			return fmt.Errorf("write stdout: %w", err)
		}
	}
	return nil
}

func buildHookURL(rawURL, token, event, cli string) (string, error) {
	u, err := url.Parse(strings.TrimSpace(rawURL))
	if err != nil {
		return "", fmt.Errorf("invalid bridge URL: %w", err)
	}
	if u.Scheme == "" || u.Host == "" {
		return "", fmt.Errorf("invalid bridge URL %q: scheme and host required", rawURL)
	}

	token = strings.TrimSpace(token)
	event = strings.TrimSpace(event)
	cli = strings.TrimSpace(cli)

	cleanPath := strings.TrimRight(u.Path, "/")
	switch {
	case cleanPath == "" || cleanPath == "/hook":
		if event == "" {
			return "", errors.New("missing hook event (--event or XETRA_HOOK_EVENT)")
		}
		if token != "" {
			u.Path = "/hook/" + url.PathEscape(token) + "/" + url.PathEscape(event)
		} else {
			u.Path = "/hook/" + url.PathEscape(event)
		}
	case strings.HasPrefix(cleanPath, "/hook/"):
		rest := strings.TrimPrefix(cleanPath, "/hook/")
		parts := strings.Split(rest, "/")
		if len(parts) == 1 && event != "" && normalizeEvent(parts[0]) == "" {
			u.Path = cleanPath + "/" + url.PathEscape(event)
		}
	}

	if cli != "" {
		q := u.Query()
		if q.Get("cli") == "" {
			q.Set("cli", cli)
			u.RawQuery = q.Encode()
		}
	}
	return u.String(), nil
}

// RunStdioBridge implements the `artex mcp-bridge` subcommand: it bridges
// newline-delimited JSON-RPC MCP messages on stdin/stdout to the local
// Streamable HTTP MCP server endpoint, bypassing any HTTP_PROXY/HTTPS_PROXY.
func RunStdioBridge(ctx context.Context, args []string, stdin io.Reader, stdout io.Writer) error {
	fs := flag.NewFlagSet("mcp-bridge", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	bridgeURL := fs.String("url", os.Getenv("XETRA_BRIDGE_URL"), "bridge base or MCP URL")
	token := fs.String("token", os.Getenv("XETRA_BRIDGE_TOKEN"), "bridge run token")
	if err := fs.Parse(args); err != nil {
		return err
	}

	if strings.TrimSpace(*bridgeURL) == "" {
		return errors.New("missing bridge URL (--url or XETRA_BRIDGE_URL)")
	}
	targetURL, err := buildMCPURL(*bridgeURL, *token)
	if err != nil {
		return err
	}

	client := loopbackHTTPClient(0)
	tok := strings.TrimSpace(*token)

	var (
		outMu     sync.Mutex
		sessMu    sync.Mutex
		sessionID string
		wg        sync.WaitGroup
	)

	writeLine := func(data []byte) {
		trimmed := bytes.TrimSpace(data)
		if len(trimmed) == 0 {
			return
		}
		outMu.Lock()
		_, _ = stdout.Write(append(trimmed, '\n'))
		outMu.Unlock()
	}

	reader := bufio.NewReaderSize(stdin, 64*1024)
	for {
		line, readErr := reader.ReadBytes('\n')
		trimmed := bytes.TrimSpace(line)
		if len(trimmed) > 0 {
			msg := append([]byte(nil), trimmed...)
			wg.Add(1)
			go func() {
				defer wg.Done()
				forwardStdioMCPMessage(ctx, client, targetURL, tok, &sessMu, &sessionID, msg, writeLine)
			}()
		}
		if readErr != nil {
			if errors.Is(readErr, io.EOF) {
				break
			}
			wg.Wait()
			return fmt.Errorf("read stdin: %w", readErr)
		}
	}
	wg.Wait()
	return nil
}

func buildMCPURL(rawURL, token string) (string, error) {
	u, err := url.Parse(strings.TrimSpace(rawURL))
	if err != nil {
		return "", fmt.Errorf("invalid bridge URL: %w", err)
	}
	if u.Scheme == "" || u.Host == "" {
		return "", fmt.Errorf("invalid bridge URL %q: scheme and host required", rawURL)
	}
	token = strings.TrimSpace(token)
	cleanPath := strings.TrimRight(u.Path, "/")
	if (cleanPath == "" || cleanPath == "/mcp") && token != "" {
		u.Path = "/mcp/" + url.PathEscape(token)
	}
	return u.String(), nil
}

func forwardStdioMCPMessage(
	ctx context.Context,
	client *http.Client,
	targetURL string,
	token string,
	sessMu *sync.Mutex,
	sessionID *string,
	msg []byte,
	writeLine func([]byte),
) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, targetURL, bytes.NewReader(msg))
	if err != nil {
		writeJSONRPCFallbackError(msg, err.Error(), writeLine)
		return
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	if token != "" {
		req.Header.Set("X-Xetra-Bridge-Token", token)
	}
	sessMu.Lock()
	sid := *sessionID
	sessMu.Unlock()
	if sid != "" {
		req.Header.Set("Mcp-Session-Id", sid)
	}

	resp, err := client.Do(req)
	if err != nil {
		writeJSONRPCFallbackError(msg, err.Error(), writeLine)
		return
	}
	defer resp.Body.Close()

	if newSID := strings.TrimSpace(resp.Header.Get("Mcp-Session-Id")); newSID != "" {
		sessMu.Lock()
		*sessionID = newSID
		sessMu.Unlock()
	}

	if resp.StatusCode == http.StatusAccepted || resp.StatusCode == http.StatusNoContent {
		_, _ = io.Copy(io.Discard, resp.Body)
		return
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		writeJSONRPCFallbackError(msg, fmt.Sprintf("HTTP %d: %s", resp.StatusCode, strings.TrimSpace(string(body))), writeLine)
		return
	}

	ct := strings.ToLower(resp.Header.Get("Content-Type"))
	if strings.Contains(ct, "text/event-stream") {
		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 64*1024), maxBodyBytes)
		for scanner.Scan() {
			line := scanner.Text()
			if strings.HasPrefix(line, "data:") {
				data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
				if data != "" {
					writeLine([]byte(data))
				}
			}
		}
		return
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxBodyBytes))
	if err != nil {
		writeJSONRPCFallbackError(msg, err.Error(), writeLine)
		return
	}
	if len(bytes.TrimSpace(body)) > 0 {
		var compacted bytes.Buffer
		if json.Compact(&compacted, body) == nil {
			writeLine(compacted.Bytes())
		} else {
			writeLine(body)
		}
	}
}

func writeJSONRPCFallbackError(reqBytes []byte, errMsg string, writeLine func([]byte)) {
	var probe struct {
		ID json.RawMessage `json:"id"`
	}
	if json.Unmarshal(reqBytes, &probe) != nil {
		return
	}
	trimmedID := bytes.TrimSpace(probe.ID)
	if len(trimmedID) == 0 || bytes.Equal(trimmedID, []byte("null")) {
		return // notification; no response expected
	}
	out, err := json.Marshal(rpcResponse{
		JSONRPC: "2.0",
		ID:      probe.ID,
		Error: &rpcError{
			Code:    -32603,
			Message: errMsg,
		},
	})
	if err == nil {
		writeLine(out)
	}
}
