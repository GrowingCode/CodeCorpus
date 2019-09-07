/**
 * @jdk版本:1.6
 * @编码时间:2010-3-20
 */
package com.mingrisoft.SAX_demo;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;

import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

import com.mingrisoft.entity.Book;
import com.mingrisoft.entity.BookPrice;

/**
 * @author bwm
 * 
 */
public class BookSAXParsing extends DefaultHandler {

	private List<Book> bookList;

	private Book book;

	private BookPrice bookPrice;

	private String elementValue;

	@Override
	public void startDocument() throws SAXException {
		// TODO Auto-generated method stub
		bookList = new ArrayList<Book>();
	}

	/**
	 * 读取元素、属性，把元素、属性名称保存在list中
	 */
	@Override
	public void startElement(String uri, String localName, String qName,
			Attributes attributes) throws SAXException {
		// TODO Auto-generated method stub

		if (book == null || "book".equals(localName)) {
			book = new Book();
		}

		if (bookPrice == null || "price".equals(localName)) {
			bookPrice = new BookPrice();
			// 读取属性名称和值
			Map<String, String> attributeMap = new HashMap<String, String>();
			for (int i = 0; i < attributes.getLength(); i++) {
				attributeMap.put(attributes.getLocalName(i),
						attributes.getValue(i));
			}
			if (!attributeMap.isEmpty()) {
				bookPrice.setAttributeMap(attributeMap);
			}

		}
	}

	@Override
	public void endElement(String uri, String localName, String qName)
			throws SAXException {
		// TODO Auto-generated method stub

		if ("name".equals(localName)) {
			book.setName(elementValue);
		} else if ("publisher".equals(localName)) {
			book.setPublisher(elementValue);
		} else if ("company".equals(localName)) {
			book.setCompany(elementValue);
		} else if ("author".equals(localName)) {
			book.setAuthor(elementValue);
		} else if ("ISBN".equals(localName)) {
			book.setIsbn(elementValue);
		} else if ("price".equals(localName)) {
			bookPrice.setValue(Double.valueOf(elementValue));
			book.setPrice(bookPrice);
		} else if ("url".equals(localName)) {
			try {
				book.setUrl(new URL(elementValue));
			} catch (MalformedURLException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		} else if ("book".equals(localName)) {
			bookList.add(book);
		}
	}

	/**
	 * 解析元素内容
	 */
	@Override
	public void characters(char[] ch, int start, int length)
			throws SAXException {
		// TODO Auto-generated method stub
		elementValue = elementValue.valueOf(ch, start, length);
		elementValue = elementValue.replace("\t", "");
		elementValue = elementValue.replace(" ", "");
		elementValue = elementValue.replace("\n", "");
		elementValue = elementValue.replace("\r", "");
	}

	public List<Book> getBookList() {
		return this.bookList;
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
		BookSAXParsing elementSAXParsing = new BookSAXParsing();
		elementSAXParsing.parseReadFile(pathname);
		System.out.println("自定义JAVA类封装元素名称和内容");
		System.out.println(elementSAXParsing.getBookList());
	}

}
