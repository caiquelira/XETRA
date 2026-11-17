//go:build !windows

package agent

import (
	"errors"
	"os/exec"
	"sync"
	"syscall"
	"time"
)

const unixTermGracePeriod = 200 * time.Millisecond

type platformProcTree struct {
	mu   sync.Mutex
	pgid int
}

// preparePlatformProcTree places the child into its own process group so
// negative-PGID signalling reaches all descendant processes.
func preparePlatformProcTree(cmd *exec.Cmd) (*platformProcTree, error) {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.Setpgid = true
	return &platformProcTree{}, nil
}

func (p *platformProcTree) attach(pid int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.pgid = pid
}

func (p *platformProcTree) kill(cmd *exec.Cmd, done <-chan struct{}) error {
	p.mu.Lock()
	pgid := p.pgid
	p.mu.Unlock()

	if pgid <= 0 {
		if cmd != nil && cmd.Process != nil {
			return cmd.Process.Kill()
		}
		return nil
	}

	if err := syscall.Kill(-pgid, syscall.SIGTERM); err != nil && errors.Is(err, syscall.ESRCH) {
		return nil
	}

	timer := time.NewTimer(unixTermGracePeriod)
	defer timer.Stop()

	select {
	case <-done:
		return nil
	case <-timer.C:
		if err := syscall.Kill(-pgid, syscall.SIGKILL); err != nil && !errors.Is(err, syscall.ESRCH) {
			return err
		}
		return nil
	}
}

func (p *platformProcTree) cleanup() {}

func processAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	err := syscall.Kill(pid, 0)
	return err == nil || errors.Is(err, syscall.EPERM)
}
