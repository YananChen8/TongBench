#pragma once

#include "rpc_event.h"
#include "rpc_reactor.h"
#include "rpc_responder.h"
#include "rpc_type.h"
#include <functional>
#include <optional>
#include <shared_mutex>

namespace tongos
{
	class RpcRouter : public RpcTypeQueryer
	{
	public:
		RpcRouter()
		{
		}

		using RouteHandlerPtr_ = void(RpcEvent rpc_message);
		using RouteHandler = std::function<RouteHandlerPtr_>;
		// todo 挪到private里
		using RpcReactorGenerator = std::function<RpcReactorBase *()>;

		struct RpcRoute
		{
			RpcType rpc_type;
			RpcReactorGenerator rpc_reactor_generator;
		};

		template <typename Reactor>
		void registerReactorCreator(const std::string& method,
		                            std::function<Reactor *()> reactor_creator)
		{
			std::scoped_lock lock(mu);
			route_map[method] = RpcRoute{
				Reactor::rpc_type,
				reactor_creator,
			};
		}

		void registerReactorCreator(const std::string& method, RpcType rpc_type, std::function<RpcReactorBase *()> reactor_creator)
		{
			std::scoped_lock lock(mu);
			route_map[method] = RpcRoute{
				rpc_type,
				reactor_creator,
			};
		}

		template <typename Reactor>
		void registerReactorCreator(const std::string& method,
		                            Reactor*(reactor_creator)())
		{
			registerReactorCreator(method, std::function(reactor_creator));
		}

		template <typename T>
		static T* allocate() { return new T{}; }

		template <typename Reactor>
		void registerReactor(const std::string& method)
		{
			registerReactorCreator(method, allocate<Reactor>);
		}

		template <typename Request, typename Response>
		void
		registerUnary(const std::string& method,
		              std::function<ResponseStatus(Request&, Response&)> handler)
		{
			using Reactor = RpcReactorUnarySyncHandler<Request, Response>;
			registerReactorCreator<Reactor>(
				method, [handler]() { return Reactor::create(handler); });
		}

		template <typename Handler>
		void registerUnaryHandler(const std::string& method, Handler handler)
		{
			registerUnary(method, std::function(handler));
		}

		template <typename Request, typename Response>
		void registerServerStreaming(const std::string& method,
		                             std::function<void(Request& request,
		                                                RpcServerStreamingResponder<Response> responder)>
		                             handler)
		{
			using Reactor = RpcReactorServerStreamingHandler<Request, Response>;
			registerReactorCreator<Reactor>(
				method, [handler]() { return Reactor::create(handler); });
		}

		template <typename Handler>
		void registerServerStreamingHandler(const std::string& method, Handler handler)
		{
			registerServerStreaming(method, std::function(handler));
		}

		template <class Object, class Request, class Response>
		using RpcUnaryObjectHandler = ResponseStatus (Object::*)(Request&, Response&);

		std::optional<const RpcRoute*> queryRoute(const std::string& method)
		{
			std::shared_lock lock{mu};
			auto iter = route_map.find(method);
			if (iter != route_map.end())
			{
				return &iter->second;
			}
			else
			{
				return {};
			}
		}

		std::optional<RpcType> queryRpcType(const std::string& method)
		{
			auto optional_route = queryRoute(method);
			if (optional_route)
			{
				return optional_route.value()->rpc_type;
			}
			else
			{
				return {};
			}
		}

		void handle(RpcEvent rpc_event)
		{
			RpcReactorBase* rpc_reacotr = rpc_event.backuped_rpc_reactor();
			RpcStream* rpc_stream = rpc_event.rpc_stream();
			if (rpc_reacotr == nullptr && rpc_stream != nullptr)
			{
				// cancel事件可能在request还没来得及处理就触发了，所以如果读到的reactor是nullptr，需要从rpc_stream再读取一次
				rpc_reacotr = rpc_stream->getRpcReactor();
			}

			if (rpc_reacotr == nullptr)
			{
				if (rpc_event.event_type() > RpcEventType::REQUEST)
				{
					// 除了未注册的方法，这种情况不可能出现
					return;
				}

				auto optional_route = queryRoute(rpc_stream->method());
				if (!optional_route)
				{
					// 方法未注册
					rpc_stream->finish(ResponseStatus(grpc::UNIMPLEMENTED, "unknown method"));
					UE_LOG(LogTongSimGRPC, Error, TEXT("[handle] Received request for unknown method: %s"), UTF8_TO_TCHAR(rpc_stream->method().c_str()));
					return;
				}
				// 创建reactor
				rpc_reacotr = optional_route.value()->rpc_reactor_generator();
				rpc_stream->bindRpcReactor(rpc_reacotr);
				rpc_reacotr->bindRpcStream(rpc_event);
				rpc_reacotr->init();

				UE_LOG(LogTongSimGRPC, Log, TEXT("[handle] Created new RpcReactor for method: %s"), UTF8_TO_TCHAR(rpc_stream->method().c_str()));
			}

			// 处理消息
			try
			{
				rpc_reacotr->handle(rpc_event);
			}
			catch (const RpcException& ex)
			{
				ResponseStatus status = ex.status();
				UE_LOG(LogTongSimGRPC, Error, TEXT("[handle] RpcReactor threw RpcException: code = %d, message = %s"),
				       static_cast<int32>(status.error_code()), UTF8_TO_TCHAR(status.error_message().c_str()));
				rpc_reacotr->finish(status);
			}
			catch (std::exception& e)
			{
				UE_LOG(LogTongSimGRPC, Error, TEXT("[handle] RpcReactor threw std::exception: %s"), UTF8_TO_TCHAR(e.what()));
				rpc_reacotr->finish({grpc::StatusCode::INTERNAL, e.what()});
			}
			catch (...)
			{
				UE_LOG(LogTongSimGRPC, Error, TEXT("[handle] RpcReactor threw unknown exception"));
				rpc_reacotr->finish({
					grpc::StatusCode::INTERNAL,
					"reactor handler throwed unknown exception!"
				});
			}
		}

	private:
		// todo 这里的Key类型应该是什么
		using RouteMap = std::map<std::string, RpcRoute, std::less<>>;
		std::shared_mutex mu;
		RouteMap route_map;
	};
} // namespace tongos
