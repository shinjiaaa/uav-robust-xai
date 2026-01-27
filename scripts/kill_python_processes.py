"""Kill all Python processes to free locked model files."""

import sys
import subprocess
import os
import psutil

def main():
    """Kill all Python processes except the current one."""
    current_pid = os.getpid()
    
    if sys.platform == 'win32':
        try:
            # Use psutil for more reliable process management
            python_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
                try:
                    if 'python' in proc.info['name'].lower():
                        if proc.info['pid'] != current_pid:
                            python_processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            if not python_processes:
                print("No other Python processes found.")
                return
            
            print(f"Found {len(python_processes)} Python processes:")
            for proc in python_processes:
                try:
                    mem_mb = proc.info['memory_info'].rss / 1024 / 1024
                    print(f"  PID {proc.info['pid']}: {mem_mb:.1f} MB")
                except:
                    print(f"  PID {proc.info['pid']}")
            
            print(f"\nKilling {len(python_processes)} Python processes...")
            killed = 0
            for proc in python_processes:
                try:
                    proc.terminate()  # Graceful termination first
                    proc.wait(timeout=3)
                    killed += 1
                    print(f"  [OK] Killed PID {proc.info['pid']}")
                except psutil.TimeoutExpired:
                    try:
                        proc.kill()  # Force kill if needed
                        killed += 1
                        print(f"  [OK] Force killed PID {proc.info['pid']}")
                    except Exception as e:
                        print(f"  [FAIL] Could not kill PID {proc.info['pid']}: {e}")
                except Exception as e:
                    print(f"  [FAIL] Could not kill PID {proc.info['pid']}: {e}")
            
            print(f"\n[OK] Killed {killed}/{len(python_processes)} Python processes.")
            print("[OK] Model files should now be accessible.")
            
        except ImportError:
            # Fallback to taskkill if psutil not available
            print("psutil not available, using taskkill...")
            try:
                result = subprocess.run(
                    ['taskkill', '/F', '/IM', 'python.exe'],
                    capture_output=True,
                    text=True,
                    check=False
                )
                print(result.stdout)
                if "No tasks" not in result.stdout:
                    print("[OK] Python processes killed.")
            except Exception as e:
                print(f"Error: {e}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print("This script only works on Windows. Use 'pkill python' on Linux/Mac.")


if __name__ == "__main__":
    main()
