package agent

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// CLISessionRecord persists the mapping between a XETRA session key (such as
// "exp12-worker-i3" or "conv-42") and the underlying CLI's native session or
// conversation identifier.
type CLISessionRecord struct {
	SessionKey string    `json:"session_key"`
	Format     string    `json:"format,omitempty"`
	SessionID  string    `json:"session_id"`
	UpdatedAt  time.Time `json:"updated_at"`
}

// CLISessionStore atomically persists CLI session mappings under a directory
// as `<sanitized-key>.json` files.
type CLISessionStore struct {
	dir string
	mu  sync.Mutex
}

// NewCLISessionStore returns a CLISessionStore backed by dir.
func NewCLISessionStore(dir string) *CLISessionStore {
	return &CLISessionStore{dir: strings.TrimSpace(dir)}
}

// Dir returns the backing directory path of the store.
func (s *CLISessionStore) Dir() string {
	if s == nil {
		return ""
	}
	return s.dir
}

// Load reads the persisted session record for key. Missing entries return
// ok=false, err=nil; unreadable or corrupt files return a non-nil error.
func (s *CLISessionStore) Load(key string) (CLISessionRecord, bool, error) {
	if s == nil || s.dir == "" {
		return CLISessionRecord{}, false, fmt.Errorf("cli session store directory is empty")
	}
	if strings.TrimSpace(key) == "" {
		return CLISessionRecord{}, false, nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	data, err := os.ReadFile(s.filePath(key))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return CLISessionRecord{}, false, nil
		}
		return CLISessionRecord{}, false, fmt.Errorf("read cli session %q: %w", key, err)
	}
	var rec CLISessionRecord
	if err := json.Unmarshal(data, &rec); err != nil {
		return CLISessionRecord{}, false, fmt.Errorf("decode cli session %q: %w", key, err)
	}
	if rec.SessionID == "" {
		return rec, false, nil
	}
	return rec, true, nil
}

// Save atomically writes rec under key using a temp file and rename.
func (s *CLISessionStore) Save(key string, rec CLISessionRecord) error {
	if s == nil || s.dir == "" {
		return fmt.Errorf("cli session store directory is empty")
	}
	if strings.TrimSpace(key) == "" {
		return fmt.Errorf("cli session key is empty")
	}
	if strings.TrimSpace(rec.SessionID) == "" {
		return fmt.Errorf("cli session id is empty")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if err := os.MkdirAll(s.dir, 0o755); err != nil {
		return fmt.Errorf("create cli session dir %q: %w", s.dir, err)
	}

	rec.SessionKey = key
	if rec.UpdatedAt.IsZero() {
		rec.UpdatedAt = time.Now().UTC()
	}
	payload, err := json.MarshalIndent(rec, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal cli session %q: %w", key, err)
	}
	payload = append(payload, '\n')

	tmp, err := os.CreateTemp(s.dir, ".session-*.tmp")
	if err != nil {
		return fmt.Errorf("create temp file for cli session %q: %w", key, err)
	}
	tmpPath := tmp.Name()
	cleanupTmp := true
	defer func() {
		if cleanupTmp {
			_ = os.Remove(tmpPath)
		}
	}()

	if _, err := tmp.Write(payload); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("write temp file for cli session %q: %w", key, err)
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("sync temp file for cli session %q: %w", key, err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close temp file for cli session %q: %w", key, err)
	}

	target := s.filePath(key)
	if err := os.Rename(tmpPath, target); err != nil {
		return fmt.Errorf("rename cli session %q to %q: %w", key, target, err)
	}
	cleanupTmp = false
	return nil
}

// Delete removes any persisted session record for key. Missing entries are
// treated as a no-op.
func (s *CLISessionStore) Delete(key string) error {
	if s == nil || s.dir == "" {
		return fmt.Errorf("cli session store directory is empty")
	}
	if strings.TrimSpace(key) == "" {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	err := os.Remove(s.filePath(key))
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("delete cli session %q: %w", key, err)
	}
	return nil
}

func (s *CLISessionStore) filePath(key string) string {
	return filepath.Join(s.dir, sanitizeCLISessionKey(key)+".json")
}

// sanitizeCLISessionKey produces a cross-platform, traversal-free filename stem
// from key. Normal keys (alphanumeric, '.', '_', '-' without '..') stay
// verbatim; anything needing sanitization gets a short SHA-256 suffix so two
// distinct keys never collide.
func sanitizeCLISessionKey(key string) string {
	var b strings.Builder
	b.Grow(len(key))
	modified := false
	for _, r := range key {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' || r == '_' || r == '.' {
			b.WriteRune(r)
		} else {
			b.WriteByte('_')
			modified = true
		}
	}
	sanitized := b.String()
	if sanitized == "" || sanitized == "." || sanitized == ".." || strings.Contains(sanitized, "..") || strings.HasPrefix(sanitized, ".") {
		sanitized = strings.ReplaceAll(sanitized, "..", "__")
		sanitized = strings.TrimLeft(sanitized, ".")
		if sanitized == "" {
			sanitized = "session"
		}
		modified = true
	}
	// Bound length safely below common 255-byte filesystem filename limits.
	const maxStem = 120
	if len(sanitized) > maxStem {
		sanitized = sanitized[:maxStem]
		modified = true
	}
	if modified {
		sum := sha256.Sum256([]byte(key))
		sanitized += "-" + hex.EncodeToString(sum[:4])
	}
	return sanitized
}
