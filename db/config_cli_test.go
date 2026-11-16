package db

import (
	"fmt"
	"regexp"
	"strings"
	"testing"
	"time"
)

func TestCLIIsKeylessProfileFormat(t *testing.T) {
	cases := []struct {
		format string
		want   bool
	}{
		{format: "ollama", want: true},
		{format: "claude-cli", want: true},
		{format: "agy-cli", want: true},
		{format: "  CLAUDE-CLI  ", want: true},
		{format: "openai", want: false},
		{format: "anthropic", want: false},
		{format: "openai-responses", want: false},
		{format: "", want: false},
		{format: "unknown", want: false},
	}
	for _, tc := range cases {
		if got := IsKeylessProfileFormat(tc.format); got != tc.want {
			t.Errorf("IsKeylessProfileFormat(%q) = %v, want %v", tc.format, got, tc.want)
		}
	}
}

func TestCLISchemaConstraints(t *testing.T) {
	re := regexp.MustCompile(`CHECK\s*\(\s*format\s+IN\s*\([^)]+\)\s*\)`)
	matches := re.FindAllString(schemaSQL, -1)
	if len(matches) != 2 {
		t.Fatalf("found %d llm_profiles format CHECK constraints in schema.sql, want 2 (CREATE TABLE + ALTER TABLE)", len(matches))
	}
	required := []string{"'openai'", "'anthropic'", "'openai-responses'", "'ollama'", "'claude-cli'", "'agy-cli'"}
	for i, constraint := range matches {
		for _, tok := range required {
			if !strings.Contains(constraint, tok) {
				t.Errorf("constraint #%d (%s) missing %s", i+1, constraint, tok)
			}
		}
	}
}

func TestCLIProfilesKeylessCRUDAndPool(t *testing.T) {
	d, err := Open(testDSN(t))
	if err != nil {
		t.Skipf("postgres unavailable (%v) — skipping", err)
	}
	t.Cleanup(func() { d.Close() })

	sfx := fmt.Sprintf("-%d", time.Now().UnixNano())
	mk := func(name, format, model, baseURL string, priority int, exclude bool) int64 {
		id, err := d.SaveProfile(&LLMProfile{
			Name:        name + sfx,
			Format:      format,
			Model:       model,
			BaseURL:     baseURL,
			Priority:    priority,
			PoolExclude: exclude,
			Streaming:   true,
		})
		if err != nil {
			t.Fatalf("SaveProfile(%s): %v", format, err)
		}
		t.Cleanup(func() { deleteTestProfile(t, d, id) })
		return id
	}

	claudeID := mk("t-cli-claude", "claude-cli", "sonnet", "/usr/local/bin/claude", 80, false)
	agyID := mk("t-cli-agy", "agy-cli", "gemini-4-hydrogen-high", "", 60, false)
	excludedID := mk("t-cli-excluded", "claude-cli", "opus", "", 99, true)

	for _, tc := range []struct {
		id      int64
		format  string
		model   string
		baseURL string
	}{
		{id: claudeID, format: "claude-cli", model: "sonnet", baseURL: "/usr/local/bin/claude"},
		{id: agyID, format: "agy-cli", model: "gemini-4-hydrogen-high", baseURL: ""},
	} {
		got, err := d.ProfileByID(tc.id)
		if err != nil {
			t.Fatalf("ProfileByID(%d): %v", tc.id, err)
		}
		if got == nil {
			t.Fatalf("ProfileByID(%d) = nil", tc.id)
		}
		if got.Format != tc.format || got.Model != tc.model || got.BaseURL != tc.baseURL || got.APIKey != "" {
			t.Fatalf("ProfileByID(%d) = %+v, want format=%q model=%q baseURL=%q apiKey=\"\"", tc.id, got, tc.format, tc.model, tc.baseURL)
		}
	}

	list, err := d.ListProfiles()
	if err != nil {
		t.Fatalf("ListProfiles: %v", err)
	}
	found := map[int64]bool{}
	for _, p := range list {
		if p.ID == claudeID || p.ID == agyID {
			found[p.ID] = true
		}
	}
	if !found[claudeID] || !found[agyID] {
		t.Fatalf("ListProfiles missing keyless CLI profiles: found=%v", found)
	}

	chain, err := d.PoolProfiles()
	if err != nil {
		t.Fatalf("PoolProfiles: %v", err)
	}
	var gotChain []int64
	for _, p := range chain {
		switch p.ID {
		case claudeID, agyID:
			gotChain = append(gotChain, p.ID)
		case excludedID:
			t.Fatal("excluded keyless CLI profile entered the failover chain")
		}
	}
	if len(gotChain) != 2 || gotChain[0] != claudeID || gotChain[1] != agyID {
		t.Fatalf("PoolProfiles order for keyless CLI profiles = %v, want [%d %d]", gotChain, claudeID, agyID)
	}
}
