#ifndef GUNDAM_WINDOWS_PROCESS_H
#define GUNDAM_WINDOWS_PROCESS_H

#if defined(_WIN32)
#include "GenericToolbox.Windows.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace GundamUtils {
  struct WindowsHandleCloser {
    void operator()(HANDLE handle_) const {
      if( handle_ != nullptr and handle_ != INVALID_HANDLE_VALUE ){ CloseHandle(handle_); }
    }
  };
  using WindowsHandle = std::unique_ptr<void, WindowsHandleCloser>;

  // A child process with binary stdin/stdout pipes. Arguments go directly to
  // CreateProcess, without a command shell. Closing the object closes handles;
  // callers choose when to close stdin and wait for the child's exit.
  class WindowsProcess {
  public:
    explicit WindowsProcess(const std::vector<std::string>& arguments_, bool captureErrors_ = false){
      if( arguments_.empty() ){ throw std::invalid_argument("Missing process executable"); }
      SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
      HANDLE readHandle{}, writeHandle{};
      if( not CreatePipe(&readHandle, &writeHandle, &security, 0) ){ fail("CreatePipe(stdin)"); }
      WindowsHandle childInput{readHandle};
      _input_.reset(writeHandle);
      if( not SetHandleInformation(_input_.get(), HANDLE_FLAG_INHERIT, 0) ){ fail("SetHandleInformation(stdin)"); }
      if( not CreatePipe(&readHandle, &writeHandle, &security, 0) ){ fail("CreatePipe(stdout)"); }
      _output_.reset(readHandle);
      WindowsHandle childOutput{writeHandle};
      if( not SetHandleInformation(_output_.get(), HANDLE_FLAG_INHERIT, 0) ){ fail("SetHandleInformation(stdout)"); }

      WindowsHandle childError;
      if( not captureErrors_ ){
        HANDLE errorHandle{};
        const HANDLE parentError = GetStdHandle(STD_ERROR_HANDLE);
        if( parentError != nullptr and parentError != INVALID_HANDLE_VALUE ){
          if( not DuplicateHandle(GetCurrentProcess(), parentError, GetCurrentProcess(),
                                  &errorHandle, 0, TRUE, DUPLICATE_SAME_ACCESS) ){ fail("DuplicateHandle(stderr)"); }
        }
        else{
          errorHandle = CreateFileA("NUL", GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                                    &security, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
          if( errorHandle == INVALID_HANDLE_VALUE ){ fail("CreateFile(stderr)"); }
        }
        childError.reset(errorHandle);
      }

      STARTUPINFOEXA startup{};
      startup.StartupInfo.cb = sizeof(startup);
      startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
      startup.StartupInfo.hStdInput = childInput.get();
      startup.StartupInfo.hStdOutput = childOutput.get();
      startup.StartupInfo.hStdError = captureErrors_ ? childOutput.get() : childError.get();
      // Inherit only these handles, including when several workers start at once.
      SIZE_T attributeSize{};
      InitializeProcThreadAttributeList(nullptr, 1, 0, &attributeSize);
      std::vector<unsigned char> attributes(attributeSize);
      startup.lpAttributeList = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attributes.data());
      if( not InitializeProcThreadAttributeList(startup.lpAttributeList, 1, 0, &attributeSize) ){
        fail("InitializeProcThreadAttributeList");
      }
      struct AttributeGuard {
        LPPROC_THREAD_ATTRIBUTE_LIST list;
        ~AttributeGuard(){ DeleteProcThreadAttributeList(list); }
      } attributeGuard{startup.lpAttributeList};
      HANDLE inheritedHandles[]{childInput.get(), childOutput.get(), startup.StartupInfo.hStdError};
      const SIZE_T inheritedSize = captureErrors_ ? 2*sizeof(HANDLE) : sizeof(inheritedHandles);
      if( not UpdateProcThreadAttribute(startup.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                                       inheritedHandles, inheritedSize, nullptr, nullptr) ){
        fail("UpdateProcThreadAttribute");
      }

      std::string commandLine;
      for( const auto& argument : arguments_ ){
        if( not commandLine.empty() ){ commandLine += ' '; }
        commandLine += quoteArgument(argument);
      }
      PROCESS_INFORMATION process{};
      if( not CreateProcessA(arguments_.front().c_str(), commandLine.data(), nullptr, nullptr, TRUE,
                             CREATE_NO_WINDOW | EXTENDED_STARTUPINFO_PRESENT, nullptr, nullptr,
                             &startup.StartupInfo, &process) ){ fail("CreateProcess"); }
      _process_.reset(process.hProcess);
      WindowsHandle thread{process.hThread};
    }

    std::size_t read(char* buffer_, std::size_t size_){
      DWORD count{};
      if( not ReadFile(_output_.get(), buffer_, transferSize(size_), &count, nullptr) ){
        if( GetLastError() == ERROR_BROKEN_PIPE ){ return 0; }
        fail("ReadFile(child stdout)");
      }
      return count;
    }
    std::size_t write(const char* buffer_, std::size_t size_){
      DWORD count{};
      if( not WriteFile(_input_.get(), buffer_, transferSize(size_), &count, nullptr) ){ fail("WriteFile(child stdin)"); }
      return count;
    }
    void closeInput(){ _input_.reset(); }
    DWORD wait(){
      if( WaitForSingleObject(_process_.get(), INFINITE) != WAIT_OBJECT_0 ){ fail("WaitForSingleObject"); }
      DWORD code{};
      if( not GetExitCodeProcess(_process_.get(), &code) ){ fail("GetExitCodeProcess"); }
      return code;
    }

  private:
    [[noreturn]] static void fail(const char* operation_){
      throw std::system_error(static_cast<int>(GetLastError()), std::system_category(), operation_);
    }
    static DWORD transferSize(std::size_t size_){
      return static_cast<DWORD>(std::min(size_, static_cast<std::size_t>(std::numeric_limits<DWORD>::max())));
    }
    static std::string quoteArgument(const std::string& argument_){
      std::string result{"\""};
      std::size_t backslashes = 0;
      for( const char character : argument_ ){
        if( character == '\\' ){ ++backslashes; continue; }
        if( character == '"' ){
          result.append(2*backslashes + 1, '\\');
        }
        else{ result.append(backslashes, '\\'); }
        result += character;
        backslashes = 0;
      }
      result.append(2*backslashes, '\\');
      result += '"';
      return result;
    }

    WindowsHandle _input_;
    WindowsHandle _output_;
    WindowsHandle _process_;
  };
}
#endif

#endif
