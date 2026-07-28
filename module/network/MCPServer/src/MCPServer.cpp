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

        server.tool("cmd",  // an exciting tool - this one will run commands on the terminal
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

                        std::string out = run(command);  // run it

                        log<DEBUG>("cmd returning:", out.c_str());  // echo the output :D

                        return {
                            .content = {mcp::TextContent{.text = "Command returned " + out}},  // return to the client
                        };
                    });
    }

}  // namespace module::network
