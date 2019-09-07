/**
 * @jdk版本:1.6
 * @编码时间:2010-3-20
 */
package com.mingrisoft.SAX_demo;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;

import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

/**
 * @author bwm
 * 
 */
public class ElementValueSAXParsing extends DefaultHandler {

	private List<String> list = new ArrayList<String>();

	private String value;

	/**
	 * 读取当前元素的内容,过滤回制表符、空格符、回车符、换行符
	 */
	@Override
	public void characters(char[] ch, int start, int length)
			throws SAXException {
		//读取当前元素的内容,过滤回制表符、空格符、回车符、换行符
		value = String.valueOf(ch, start, length);
		value = value.replace("\t", "");
		value = value.replace(" ", "");
		value = value.replace("\n", "");
		value = value.replace("\r", "");
	}

	/**
	 * 读取元素结束，把元素名称和元素内容保存在list中
	 */
	@Override
	public void endElement(String uri, String localName, String qName)
			throws SAXException {
		//读取元素结束，把元素名称和元素内容保存在list中
		list.add(localName + ":" + value);
	}

	public List<String> getList() {
		return this.list;
	}

	/**
	 * 通过文件读取XML
	 * 
	 * @param pathname
	 *            文件路径
	 */
	public void parseReadFile(String pathname) {
		SAXParser parser;
		SAXParserFactory factory = SAXParserFactory.newInstance();
		try {
			factory.setValidating(true);
			factory.setNamespaceAware(true);
			parser = factory.newSAXParser();
			File file = new File(pathname);
			parser.parse(file, this);
		} catch (ParserConfigurationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (SAXException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

	public static void main(String[] arg) {

		String pathname = "xmldemo/books.xml";
		ElementValueSAXParsing elementSAXParsing = new ElementValueSAXParsing();
		elementSAXParsing.parseReadFile(pathname);
		System.out.println("元素名称和元素内容");
		System.out.println(elementSAXParsing.getList());
	}
}
