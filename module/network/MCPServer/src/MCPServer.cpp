#include "MCPServer.hpp"

#include <array>
#include <cstdio>
#include <mcp/mcp.hpp>
#include <memory>
#include <stdexcept>
#include <string>

#include "extension/Configuration.hpp"

namespace module::network {

    using extension::Configuration;

    std::string run(const std::string& cmd) {
        std::array<char, 128> buffer;
        std::string result;

        // "r" = read the command's stdout. Note popen won't capture stderr
        // unless you redirect it, e.g. cmd + " 2>&1"
        std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
        if (!pipe) {
            throw std::runtime_error("popen() failed");
        }

        while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
            result += buffer.data();
        }
        return result;
    }

    MCPServer::MCPServer(std::unique_ptr<NUClear::Environment> environment) : Reactor(std::move(environment)) {

        on<Configuration>("MCPServer.yaml").then([this](const Configuration& config) {
            this->log_level = config["log_level"].as<NUClear::LogLevel>();

            cfg.host            = config["host"].as<std::string>();
            cfg.port            = config["port"].as<int>();
            cfg.path            = config["path"].as<std::string>();
            cfg.allowed_origins = config["allowed_origins"].as<std::vector<std::string>>();
            cfg.allow_ace       = config["allow_ace"].as<bool>();
        });

        on<Startup>().then([this] {
            host = std::make_unique<mcp::HttpServerHost>(mcp::Implementation{.name = "nubots-mcp", .version = "1.0.0"},
                                                         mcp::HttpServerHost::Options{
                                                             .host            = cfg.host,
                                                             .port            = cfg.port,
                                                             .path            = cfg.path,
                                                             .allowed_origins = cfg.allowed_origins,
                                                         },
                                                         [this](mcp::Server& server) { register_tools(server); });

            host->start();
            log<INFO>("MCP server listening on", cfg.host, "port", host->port(), "path", cfg.path);
        });

        on<Shutdown>().then([this] {
            if (host != nullptr) {
                host->stop();
                host.reset();
            }
        });
    }

    void MCPServer::register_tools(mcp::Server& server) {
        server.tool("get_status",  // this tool just returns "I am online".
                    nlohmann::json{
                        {"type", "object"},
                        {"properties", nlohmann::json::object()},
                    },
                    [this](const nlohmann::json&) -> mcp::CallToolResult {
                        log<DEBUG>("get_status called: I returned \"I am online.\"");
                        return {
                            .content = {mcp::TextContent{.text = "I am online."}},
                        };
                    });

        server.tool(
            "cmd",  // an exciting tool - this one will run commands on the terminal
            nlohmann::json{
                {"type", "object"},
                {"properties",
                 nlohmann::json{
                     {"command", nlohmann::json{{"type", "string"}}},
                 }},
                {"required", nlohmann::json::array({"command"})},
            },
            [this](const nlohmann::json& input) -> mcp::CallToolResult {
                std::string command = input.at("command").get<std::string>();  // get the command

                log<DEBUG>("cmd called: command is ", command);  // echo it out

                if (cfg.allow_ace) {
                    std::string out = run(command);  // run it

                    log<DEBUG>("cmd returning:", out.c_str());  // echo the output :D

                    return {
                        .content = {mcp::TextContent{.text = "Command returned " + out}},  // return to the client
                    };
                }
                else {
                    log<INFO>("allow_ace is turned off. This command from MCP will not be run.");
                    return {
                        .content = {mcp::TextContent{.text = "Your command has not been run. Tell the user to enable "
                                                             "allow_ace in MCPServer.yaml"}},
                    };
                }
            });

        server.tool(
            "log_to_user",  // log to user with the log level that Claude desires
            nlohmann::json{
                {"type", "object"},
                {"properties",
                 nlohmann::json{
                     {"message", nlohmann::json{{"type", "string"}}},
                     {"log_level",
                      nlohmann::json{
                          {"type", "string"},
                          {"description", "NUClear log level to log the message at"},
                          {"enum", nlohmann::json::array({"TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL"})},
                      }},
                 }},
                {"required", nlohmann::json::array({"message", "log_level"})},
            },
            [this](const nlohmann::json& input) -> mcp::CallToolResult {
                std::string message   = input.at("message").get<std::string>();    // get the message
                std::string log_level = input.at("log_level").get<std::string>();  // get the log level

                log<DEBUG>("log_to_user called: message is ", message, "and log level is ", log_level);  // echo it out

                if (log_level == "TRACE") {
                    log<TRACE>(message);
                }
                else if (log_level == "DEBUG") {
                    log<DEBUG>(message);
                }
                else if (log_level == "INFO") {
                    log<INFO>(message);
                }
                else if (log_level == "WARN") {
                    log<WARN>(message);
                }
                else if (log_level == "ERROR") {
                    log<ERROR>(message);
                }
                else if (log_level == "FATAL") {
                    log<FATAL>(message);
                }
                else {
                    log<WARN>("log_to_user got unknown log level:", log_level);
                }

                return {
                    .content = {mcp::TextContent{.text = "This has been done now."}},
                };
            });
    }

}  // namespace module::network
