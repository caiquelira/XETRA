//go:build windows

package agent

import (
	"errors"
	"os"
	"os/exec"
	"strconv"
	"sync"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

const winProcessStillActive = 259 // STILL_ACTIVE

type platformProcTree struct {
	mu  sync.Mutex
	job windows.Handle
	pid int
}

// preparePlatformProcTree configures a new Windows process group and creates a
// Job Object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE so the entire process tree
// terminates together on cancellation or handle close.
func preparePlatformProcTree(cmd *exec.Cmd) (*platformProcTree, error) {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.CreationFlags |= windows.CREATE_NEW_PROCESS_GROUP

	job, err := windows.CreateJobObject(nil, nil)
	if err != nil {
		// Degrade cleanly to taskkill /T /F fallback if Job Objects are unavailable.
		return &platformProcTree{}, nil
	}

	var info windows.JOBOBJECT_EXTENDED_LIMIT_INFORMATION
	info.BasicLimitInformation.LimitFlags = windows.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
	if _, err := windows.SetInformationJobObject(
		job,
		windows.JobObjectExtendedLimitInformation,
		uintptr(unsafe.Pointer(&info)),
		uint32(unsafe.Sizeof(info)),
	); err != nil {
		_ = windows.CloseHandle(job)
		return &platformProcTree{}, nil
	}

	return &platformProcTree{job: job}, nil
}

// attach assigns the started child process to the Job Object. If assignment
// fails (for example ERROR_ACCESS_DENIED inside a restricted parent job), the
// job handle is closed so kill falls back to taskkill /T /F /PID.
func (p *platformProcTree) attach(pid int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.pid = pid
	if p.job == 0 || pid <= 0 {
		return
	}

	procHandle, err := windows.OpenProcess(
		windows.PROCESS_SET_QUOTA|windows.PROCESS_TERMINATE,
		false,
		uint32(pid),
	)
	if err != nil {
		_ = windows.CloseHandle(p.job)
		p.job = 0
		return
	}
	defer windows.CloseHandle(procHandle)

	if err := windows.AssignProcessToJobObject(p.job, procHandle); err != nil {
		// In restricted nested-job environments without breakaway, drop the job
		// and rely on taskkill /T /F /PID during termination.
		_ = windows.CloseHandle(p.job)
		p.job = 0
	}
}

func (p *platformProcTree) kill(cmd *exec.Cmd, _ <-chan struct{}) error {
	p.mu.Lock()
	job := p.job
	p.job = 0
	pid := p.pid
	p.mu.Unlock()

	if job != 0 {
		termErr := windows.TerminateJobObject(job, 1)
		closeErr := windows.CloseHandle(job)
		if termErr == nil {
			return closeErr
		}
		if pid > 0 {
			_ = taskkillTree(pid)
		}
		if cmd != nil && cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
		return termErr
	}

	var errs []error
	if pid > 0 {
		if err := taskkillTree(pid); err != nil && processAlive(pid) {
			errs = append(errs, err)
		}
	}
	if cmd != nil && cmd.Process != nil {
		if err := cmd.Process.Kill(); err != nil && !errors.Is(err, os.ErrProcessDone) {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

func (p *platformProcTree) cleanup() {
	p.mu.Lock()
	job := p.job
	p.job = 0
	p.mu.Unlock()

	if job != 0 {
		_ = windows.CloseHandle(job)
	}
}

func taskkillTree(pid int) error {
	return exec.Command("taskkill", "/T", "/F", "/PID", strconv.Itoa(pid)).Run()
}

func processAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	h, err := windows.OpenProcess(windows.PROCESS_QUERY_LIMITED_INFORMATION, false, uint32(pid))
	if err != nil {
		return false
	}
	defer windows.CloseHandle(h)

	var code uint32
	if err := windows.GetExitCodeProcess(h, &code); err != nil {
		return false
	}
	return code == winProcessStillActive
}
