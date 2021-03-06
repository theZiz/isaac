/* This file is part of ISAAC.
 *
 * ISAAC is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * ISAAC is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with ISAAC.  If not, see <www.gnu.org/licenses/>. */

#include "Broker.hpp"
#include "WebSocketDataConnector.hpp"
#include "TCPDataConnector.hpp"
#ifdef ISAAC_SDL
	#include "SDLImageConnector.hpp"
#endif
#ifdef ISAAC_GST
	#include "RTPImageConnector.hpp"
	#include "RTMPImageConnector.hpp"
#endif
#ifdef ISAAC_JPEG
	#include "URIImageConnector.hpp"
#endif

#define ISAAC_VERSION "1.0"

#define ISAAC_INCREASE_NR_OR_DIE \
	nr++; \
	if (nr >= argc) \
	{ \
		printf("Not enough arguments!\n"); \
		return 1; \
	}

int main(int argc, char **argv)
{
	json_object_seed(0);
	int tcp_port = 2458;
	int web_port = 2459;
	int sim_port = 2460;
	const char __name[] = "ISAAC Visualization server";
	const char __url[] = "127.0.0.1";
	const char* name = __name;
	const char* url = __url;
	#ifdef ISAAC_GST
		char* twitch_apikey = NULL;
		char* twitch_url = NULL;
		int twitch_bitrate = 400;
	#endif
	int nr = 1;
	while (nr < argc)
	{
		if (strcmp(argv[nr],"--help") == 0)
		{
			printf("ISAAC - In Situ Animation of Accelerated Computations " ISAAC_VERSION"\n");
			printf("Usage:\n");
			printf("isaac [--help] [--web_port <X>] [--sim_port <X>] [--url <X>] [--name <X>]\n");
			printf("      [--jpeg] [--version]\n");
			printf("       --help: Shows this help\n");
			printf("   --web_port: Set port for the websocket clients to connect. Default %i\n",web_port);
			printf("   --tcp_port: Set port for the tcp clients to connect. Default %i\n",tcp_port);
			printf("   --sim_port: Set port for the simulations to connect to. Default %i\n",sim_port);
			printf("        --url: Set the url to connect to from outside. Default 127.0.0.1\n");
			printf("       --name: Set the name of the server.\n");
			printf("    --version: Shows the version\n");
		#ifdef ISAAC_GST
			printf("     --twitch: Set twitch apikey for twitch live streaming\n");
			printf(" --twitch_url: Set twitch rtmp-url for ssh forwarding or another rtmp service\n");
			printf("--twitch_bitrate: Set twitch bitrate. Default 400\n");
		#endif
			return 0;
		}
		else
		if (strcmp(argv[nr],"--version") == 0)
		{
			printf("Isaac version " ISAAC_VERSION "\n");
			return 0;
		}
		else
		if (strcmp(argv[nr],"--web_port") == 0)
		{
			ISAAC_INCREASE_NR_OR_DIE
			web_port = atoi(argv[nr]);
		}
		else
		if (strcmp(argv[nr],"--tcp_port") == 0)
		{
			ISAAC_INCREASE_NR_OR_DIE
			tcp_port = atoi(argv[nr]);
		}
		else
		if (strcmp(argv[nr],"--sim_port") == 0)
		{
			ISAAC_INCREASE_NR_OR_DIE
			sim_port = atoi(argv[nr]);
		}
		else
		if (strcmp(argv[nr],"--url") == 0)
		{
			ISAAC_INCREASE_NR_OR_DIE
			url = argv[nr];
		}
		else
		if (strcmp(argv[nr],"--name") == 0)
		{
			ISAAC_INCREASE_NR_OR_DIE
			name = argv[nr];
		}
		#ifdef ISAAC_GST
			else
			if (strcmp(argv[nr],"--twitch") == 0)
			{
				ISAAC_INCREASE_NR_OR_DIE
				twitch_apikey = argv[nr];
			}
			else
			if (strcmp(argv[nr],"--twitch_url") == 0)
			{
				ISAAC_INCREASE_NR_OR_DIE
				twitch_url = argv[nr];
			}
			else
			if (strcmp(argv[nr],"--twitch_bitrate") == 0)
			{
				ISAAC_INCREASE_NR_OR_DIE
				twitch_bitrate = atoi(argv[nr]);
			}
		#endif
		else
		{
			printf("Don't know argument %s\n",argv[nr]);
			return 1;
		}
		nr++;
	}
	
	printf("Using web_port=%i, tcp_port=%i and sim_port=%i\n",web_port,tcp_port,sim_port);
	
	printf("\n");
	Broker broker(name,sim_port);
	WebSocketDataConnector* webSocketDataConnector = new WebSocketDataConnector();
	if (webSocketDataConnector->init(web_port) == 0)
		broker.addDataConnector(webSocketDataConnector);
	TCPDataConnector* tCPDataConnector = new TCPDataConnector();
	if (tCPDataConnector->init(tcp_port) == 0)
		broker.addDataConnector(tCPDataConnector);
	#ifdef ISAAC_GST
		RTPImageConnector* rTPImageConnector_h264 = new RTPImageConnector(url,false,false);
		if (rTPImageConnector_h264->init(5000,5099) == 0)
			broker.addImageConnector(rTPImageConnector_h264);
		RTPImageConnector* rTPImageConnector_jpeg = new RTPImageConnector(url,false,true);
		if (rTPImageConnector_jpeg->init(5100,5199) == 0)
			broker.addImageConnector(rTPImageConnector_jpeg);
	#endif
	#ifdef ISAAC_JPEG
		URIImageConnector* uRIImageConnector = new URIImageConnector();
		if (uRIImageConnector->init(0,0) == 0)
			broker.addImageConnector(uRIImageConnector);
	#endif
	#ifdef ISAAC_GST
		RTMPImageConnector* twitchImageConnector = NULL;
		if (twitch_apikey)
		{
			if (twitch_url)
				twitchImageConnector = new RTMPImageConnector( std::string("Twitch"), std::string(twitch_apikey), std::string(twitch_url), twitch_bitrate, true );
			else
				twitchImageConnector = new RTMPImageConnector( std::string("Twitch"), std::string(twitch_apikey), std::string("live-fra.twitch.tv/app"), twitch_bitrate, true );
			if (twitchImageConnector->init(0,0) == 0)
				broker.addImageConnector(twitchImageConnector);
		}
	#endif
	#ifdef ISAAC_SDL
		SDLImageConnector* sDLImageConnector = new SDLImageConnector();
		if (sDLImageConnector->init(0,0) == 0)
			broker.addImageConnector(sDLImageConnector);
	#endif
	int return_code = 0;
	if (broker.run())
	{
		printf("Error while running isaac\n");
		return_code = -1;
	}

	delete webSocketDataConnector;
	delete tCPDataConnector;
	#ifdef ISAAC_JPEG
		delete uRIImageConnector;
	#endif
	#ifdef ISAAC_GST
		delete rTPImageConnector_h264;
		delete rTPImageConnector_jpeg;
		delete twitchImageConnector;
	#endif
	#ifdef ISAAC_SDL
		delete sDLImageConnector;
	#endif
	return return_code;
}
