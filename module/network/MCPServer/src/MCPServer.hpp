#ifndef MODULE_NETWORK_MCPSERVER_HPP
#define MODULE_NETWORK_MCPSERVER_HPP

#include <atomic>
#include <chrono>
#include <mcp/http_server_host.hpp>
#include <memory>
#include <mutex>
#include <nuclear>
#include <string>
#include <vector>

#include "message/input/Image.hpp"
#include "message/input/Sensors.hpp"
#include "message/localisation/Field.hpp"

namespace module::network {

    /// @brief Delayed trigger that stops the "walk" tool's command once its requested duration has elapsed.
    /// Carries the generation of the walk call it belongs to, so a stale delay from a superseded walk command
    /// doesn't cut a newer one short.
    struct StopWalk {
        uint64_t generation;
    };

    /// @brief Fallback trigger fired if odometry never detects movement after a "walk" call, so the duration timer
    /// still starts and the robot is not left walking forever. Carries the walk call's generation.
    struct WalkStartTimeout {
        uint64_t generation;
    };

    class MCPServer : public NUClear::Reactor {
    private:
        /// @brief Stores configuration values
        struct Config {
            /// @brief Address the MCP endpoint binds to
            std::string host = "127.0.0.1";
            /// @brief Port the MCP endpoint listens on
            int port = 8080;
            /// @brief URL path of the Streamable HTTP endpoint
            std::string path = "/mcp";
            /// @brief Origin allowlist for DNS-rebinding protection
            std::vector<std::string> allowed_origins{};
            /// @brief Boolean for allowing Claude to run random commands with no oversight
            bool allow_ace = false;
            /// @brief Torso displacement from odometry (m) after which a "walk" call counts as having started moving
            double walk_start_distance = 0.05;
            /// @brief Torso yaw change from odometry (rad) after which a "walk" call counts as having started moving
            double walk_start_yaw = 0.1;
            /// @brief Max time (s) to wait for odometry to detect movement before starting the duration timer anyway
            double walk_start_timeout = 3.0;
        } cfg;

        /// @brief The MCP Streamable HTTP host, created on Startup and stopped on Shutdown
        std::unique_ptr<mcp::HttpServerHost> host{};

        /// @brief Guards last_image, which is written from the camera reaction and read from MCP tool calls
        std::mutex image_mutex;
        /// @brief The most recent raw camera image, if one has been seen yet
        std::shared_ptr<const message::input::Image> last_image{};

        /// @brief Guards last_sensors, which is written from the Sensors reaction and read from MCP tool calls
        std::mutex sensors_mutex;
        /// @brief The most recent Sensors message if one has been seen yet
        std::shared_ptr<const message::input::Sensors> last_sensors{};

        /// @brief Guards last_field, which is written from the Field reaction and read from MCP tool calls
        std::mutex field_mutex;
        /// @brief The most recent Field localisation message if one has been seen yet
        std::shared_ptr<const message::localisation::Field> last_field{};

        /// @brief Incremented on every "walk" tool call so a delayed StopWalk can tell whether it still belongs
        /// to the walk command it was scheduled for, or whether a newer walk call has since superseded it
        std::atomic<uint64_t> walk_generation{0};

        /// @brief State of a "walk" call whose duration timer has not started yet because odometry hasn't seen the
        /// robot move. Guarded by walk_start_mutex, written from MCP tool calls and read from the Sensors reaction
        struct PendingWalkStart {
            bool active = false;
            uint64_t generation{0};
            /// @brief Requested walk duration in seconds
            double duration = 0.0;
            /// @brief When the walk was commanded, to report how long movement took to appear
            std::chrono::steady_clock::time_point commanded_at{};
            /// @brief Torso pose in world {w} when the walk was commanded
            Eigen::Isometry3d Hwt_start = Eigen::Isometry3d::Identity();
        } pending_walk_start;
        std::mutex walk_start_mutex;

        /// @brief Starts the countdown that stops walk call `generation` after `duration` seconds
        void schedule_walk_stop(uint64_t generation, double duration);

        /// @brief Register the tools exposed to each MCP session
        void register_tools(mcp::Server& server);

    public:
        /// @brief Called by the powerplant to build and setup the MCPServer reactor.
        explicit MCPServer(std::unique_ptr<NUClear::Environment> environment);
    };

}  // namespace module::network

#endif  // MODULE_NETWORK_MCPSERVER_HPP
