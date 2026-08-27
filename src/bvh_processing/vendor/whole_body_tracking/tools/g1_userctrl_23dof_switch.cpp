// Minimal G1 23-DoF user-control handoff test.
//
// Sequence:
//   WALKRUN (fsm_id 801) -> SwitchToUserCtrl()
//   hold the current pose on the 23 valid joints for a short duration
//   -> SwitchToInternalCtrl(InternalFsmMode::WALKRUN)
//
// This program deliberately does not send commands to the six joints that are
// unavailable on the 23-DoF G1 variant:
// waist_roll, waist_pitch, left/right_wrist_pitch, left/right_wrist_yaw.

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <thread>

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include "unitree/robot/g1/loco/g1_loco_api.hpp"
#include "unitree/robot/g1/loco/g1_loco_client.hpp"

namespace {

constexpr char kUserCmdTopic[] = "rt/user_lowcmd";
constexpr char kStateTopic[] = "rt/lowstate";
// Verified live on this G1 firmware after the operator entered WalkRun.
// LocoClient::Start() selects the intermediate/public start state 500; the
// actual WalkRun state reported by GetFsmId() is 801.
constexpr int kExpectedWalkRunFsmId = 801;
constexpr int kExpectedWalkRunFsmMode = 0;
constexpr float kControlDt = 0.02f;  // 50 Hz
constexpr float kKp = 20.0f;
constexpr float kKd = 1.0f;

// G1 23-DoF: 12 leg joints, waist yaw, 5 joints per arm.
constexpr std::array<int, 23> kValidJointIds = {
    0, 1, 2, 3, 4, 5,
    6, 7, 8, 9, 10, 11,
    12,
    15, 16, 17, 18, 19,
    22, 23, 24, 25, 26,
};

std::atomic_bool g_stop{false};

void OnSignal(int) { g_stop.store(true); }

struct Options {
  std::string network_interface;
  float duration_sec = 3.0f;
  bool dry_run = false;
};

bool ParseFloat(const char* text, float& value) {
  char* end = nullptr;
  value = std::strtof(text, &end);
  return end != text && end != nullptr && *end == '\0' && value >= 0.0f;
}

bool ParseArgs(int argc, char** argv, Options& options) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0]
              << " <network_interface> [duration_sec] [--dry-run]\n";
    return false;
  }
  options.network_interface = argv[1];
  if (argc >= 3 && !ParseFloat(argv[2], options.duration_sec)) {
    std::cerr << "Invalid duration_sec: " << argv[2] << '\n';
    return false;
  }
  if (argc >= 4) {
    if (std::string(argv[3]) != "--dry-run") {
      std::cerr << "Unknown option: " << argv[3] << '\n';
      return false;
    }
    options.dry_run = true;
  }
  if (argc > 4) {
    std::cerr << "Too many arguments\n";
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!ParseArgs(argc, argv, options)) {
    return 2;
  }

  std::signal(SIGINT, OnSignal);
  std::signal(SIGTERM, OnSignal);

  unitree::robot::ChannelFactory::Instance()->Init(
      0, options.network_interface);

  unitree::robot::ChannelPublisherPtr<unitree_hg::msg::dds_::LowCmd_>
      publisher;
  publisher.reset(
      new unitree::robot::ChannelPublisher<unitree_hg::msg::dds_::LowCmd_>(
          kUserCmdTopic));
  publisher->InitChannel();

  unitree_hg::msg::dds_::LowState_ latest_state;
  std::mutex state_mutex;
  std::atomic_bool state_received{false};
  unitree::robot::ChannelSubscriberPtr<unitree_hg::msg::dds_::LowState_>
      subscriber;
  subscriber.reset(
      new unitree::robot::ChannelSubscriber<unitree_hg::msg::dds_::LowState_>(
          kStateTopic));
  subscriber->InitChannel(
      [&](const void* raw_message) {
        const auto* state =
            static_cast<const unitree_hg::msg::dds_::LowState_*>(raw_message);
        {
          std::lock_guard<std::mutex> lock(state_mutex);
          latest_state = *state;
        }
        state_received.store(true);
      },
      1);

  unitree::robot::g1::LocoClient client;
  client.Init();
  client.SetTimeout(5.0f);

  int fsm_id = 0;
  const int32_t fsm_ret = client.GetFsmId(fsm_id);
  if (fsm_ret != 0 || fsm_id != kExpectedWalkRunFsmId) {
    std::cerr << "Refusing handoff: expected WALKRUN fsm_id "
              << kExpectedWalkRunFsmId << ", got " << fsm_id
              << ", ret=" << fsm_ret << '\n';
    return 3;
  }

  int fsm_mode = -1;
  const int32_t fsm_mode_ret = client.GetFsmMode(fsm_mode);
  if (fsm_mode_ret != 0 || fsm_mode != kExpectedWalkRunFsmMode) {
    std::cerr << "Refusing handoff: expected WALKRUN fsm_mode "
              << kExpectedWalkRunFsmMode << ", got " << fsm_mode
              << ", ret=" << fsm_mode_ret << '\n';
    return 3;
  }

  // Wait for a real low-state sample before constructing hold targets.
  for (int i = 0; i < 100 && !state_received.load(); ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }
  if (!state_received.load()) {
    std::cerr << "Refusing handoff: no rt/lowstate sample received\n";
    return 4;
  }

  unitree_hg::msg::dds_::LowState_ state_snapshot;
  {
    std::lock_guard<std::mutex> lock(state_mutex);
    state_snapshot = latest_state;
  }

  unitree_hg::msg::dds_::LowCmd_ command;
  for (int motor = 0; motor < 35; ++motor) {
    command.motor_cmd().at(motor).q(0.0f);
    command.motor_cmd().at(motor).dq(0.0f);
    command.motor_cmd().at(motor).kp(0.0f);
    command.motor_cmd().at(motor).kd(0.0f);
    command.motor_cmd().at(motor).tau(0.0f);
  }
  for (const int motor : kValidJointIds) {
    const float q = state_snapshot.motor_state().at(motor).q();
    if (!std::isfinite(q) || std::abs(q) > 6.5f) {
      std::cerr << "Refusing handoff: invalid joint position at motor "
                << motor << ": " << q << '\n';
      return 4;
    }
    command.motor_cmd().at(motor).q(q);
    command.motor_cmd().at(motor).dq(0.0f);
    command.motor_cmd().at(motor).kp(kKp);
    command.motor_cmd().at(motor).kd(kKd);
    command.motor_cmd().at(motor).tau(0.0f);
  }

  std::cout << "Preflight OK: fsm_id=" << fsm_id
            << ", fsm_mode=" << fsm_mode
            << ", lowstate received, 23 joint targets finite\n";
  if (options.dry_run) {
    std::cout << "Dry run complete: no rt/user_lowcmd published and no "
                 "control switch requested\n";
    return 0;
  }

  // Keep publishing in a dedicated thread so synchronous LocoClient calls can
  // never create a multi-second gap in the 50 Hz user command stream.
  std::atomic_bool publish_hold{true};
  std::thread publisher_thread([&]() {
    while (publish_hold.load()) {
      publisher->Write(command);
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
  });

  // Prime the user command topic before changing control authority.
  std::this_thread::sleep_for(std::chrono::milliseconds(200));
  if (g_stop.load()) {
    publish_hold.store(false);
    publisher_thread.join();
    std::cerr << "Interrupted before control handoff\n";
    return 5;
  }

  const int32_t switch_in_ret = client.SwitchToUserCtrl();
  if (switch_in_ret != 0) {
    std::cerr << "SwitchToUserCtrl returned " << switch_in_ret
              << "; handoff state is uncertain, keeping hold command while "
                 "requesting WALKRUN recovery\n";
    int32_t recovery_ret = -1;
    for (int attempt = 1; attempt <= 10; ++attempt) {
      recovery_ret = client.SwitchToInternalCtrl(
          unitree::robot::g1::InternalFsmMode::WALKRUN);
      if (recovery_ret == 0) {
        int recovery_fsm_id = 0;
        int32_t recovery_fsm_ret = -1;
        for (int verify = 0; verify < 20; ++verify) {
          recovery_fsm_ret = client.GetFsmId(recovery_fsm_id);
          if (recovery_fsm_ret == 0 &&
              recovery_fsm_id == kExpectedWalkRunFsmId) {
            publish_hold.store(false);
            publisher_thread.join();
            std::cerr << "Recovered to WALKRUN after failed/uncertain "
                         "handoff\n";
            return 5;
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    std::cerr << "Could not confirm WALKRUN recovery; continuing the hold "
                 "publisher for manual recovery\n";
    while (true) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
  std::cout << "User control active; holding 23 joints for "
            << options.duration_sec << " s\n";

  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::duration<float>(options.duration_sec);
  while (!g_stop.load() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  // Do not stop publishing if the return request is temporarily rejected:
  // keep the robot held while retrying the handoff.
  int32_t switch_out_ret = -1;
  for (int attempt = 1; attempt <= 50; ++attempt) {
    switch_out_ret = client.SwitchToInternalCtrl(
        unitree::robot::g1::InternalFsmMode::WALKRUN);
    if (switch_out_ret == 0) {
      break;
    }
    std::cerr << "SwitchToInternalCtrl(WALKRUN) attempt " << attempt
              << " failed: " << switch_out_ret << "; keeping hold command\n";
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  if (switch_out_ret != 0) {
    std::cerr << "Could not return to WALKRUN after retries; user command "
                 "publisher remains required for safe recovery\n";
    while (true) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  }
  publish_hold.store(false);
  publisher_thread.join();
  int returned_fsm_id = 0;
  int32_t returned_fsm_ret = -1;
  for (int attempt = 0; attempt < 20; ++attempt) {
    returned_fsm_ret = client.GetFsmId(returned_fsm_id);
    if (returned_fsm_ret == 0 &&
        returned_fsm_id == kExpectedWalkRunFsmId) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  if (returned_fsm_ret != 0 || returned_fsm_id != kExpectedWalkRunFsmId) {
    std::cerr << "WALKRUN return request succeeded, but verification got "
              << "fsm_id=" << returned_fsm_id
              << ", ret=" << returned_fsm_ret << '\n';
    return 7;
  }
  std::cout << "Returned to WALKRUN\n";
  return 0;
}
